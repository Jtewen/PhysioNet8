import os
import numpy as np
import csv
import tensorflow as tf
from tensorflow.compat.v1.summary import merge_all, scalar, FileWriter

# from matplotlib import pyplot as plt
import deep_utils
import evaluate_12ECG_score
from time import time
from tensorflow import keras
from tqdm import tqdm


def train_12ECG_classifier_transformer(input_directory, output_directory):

    tf.compat.v1.disable_eager_execution()

    logdir = "tensorboard/" + output_directory
    summary_writer = FileWriter(logdir)

    seed = 0
    np.random.seed(seed)

    epoch_size = 100

    learning_rate = 8e-4
    Length = 5600

    lambdaa_d = 1e-1

    lambdaa_DG = 1e-3

    batch_size = 64

    n_step = 160

    #######
    N_data = 10000
    dict_path = "datasets/dict_data"
    #######

    print("Loading data...")

    t0 = time()

    with open("weights.csv") as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [row for row in reader]

    class_names = data_read[0][1:]
    class_weights = np.array(data_read[1:]).astype("float32")[:, 1:]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.save(output_directory + "/class_names.npy", class_names)

    num_classes = len(class_names)

    if input_directory == "load_from_dict":

        dict_path = "../datasets/PhysioNet/dict_data"
        valid_inds = np.load(dict_path + "/valid_inds.npy")
        data_names = np.load(dict_path + "/data_names.npy")[valid_inds]
        data_ages = np.load(dict_path + "/data_ages.npy")[valid_inds]
        data_sexes = np.load(dict_path + "/data_sexes.npy")[valid_inds]
        data_labels = np.load(dict_path + "/data_labels.npy")[valid_inds]
        data_domains = np.load(dict_path + "/data_domains.npy")[valid_inds]
        index_dict = np.load(dict_path + "/index_dict.npy", allow_pickle=True).item()

        data_signals = []

        cur_i = 0
        for i in range(len(data_names)):

            name = data_names[i]
            ind = index_dict[name]
            if ind >= cur_i:

                data_dict = np.load(
                    dict_path + "/data_dict_" + str(int(ind / N_data) + 1) + ".npy",
                    allow_pickle=True,
                ).item()
                cur_i += N_data

            data_signals.append(data_dict[ind])

        assert len(data_signals) == len(data_names)
        data_dict = []

    else:
        print("Loading data from files...")
        header_files = []
        for f in tqdm(os.listdir(input_directory), "Preparing data..."):
            g = os.path.join(input_directory, f)
            if (
                not f.lower().startswith(".")
                and f.lower().endswith("hea")
                and os.path.isfile(g)
            ):
                header_files.append(g)

        num_files = len(header_files)
        print("num files", num_files)

        data_signals = []
        data_names = []
        data_ages = []
        data_sexes = []
        data_labels = []
        print("loop")
        for i in tqdm(range(num_files), "Loading data..."):
            recording, header = deep_utils.load_challenge_data(header_files[i])

            if recording.shape[0] != 12:
                print("Error in number of leads!", recording.shape)

            recording = recording.T.astype("float32")

            name = header[0].strip().split(" ")[0]
 
            try:
                samp_frq = int(header[0].strip().split(" ")[2])
            except:
                print("Error in reading sampling frequency!", header[0])
                samp_frq = 500

            if samp_frq != 500:
                recording = deep_utils.resample(recording.copy(), samp_frq)

            age = 50
            sex = 0
            label = np.zeros(num_classes, dtype="float32")

            for l in header:
                if l.startswith("# Age:"):
                    age_ = l.strip().split(" ")[2]
                    if age_ == "Nan" or age_ == "NaN":
                        age = 50
                    else:
                        try:
                            age = int(age_)
                        except:
                            print("Error in reading age!", age_)
                            age = 50

                if l.startswith("# Sex:"):
                    sex_ = l.strip().split(" ")[2]
                    if sex_ == "Male" or sex_ == "male":
                        sex = 0
                    elif sex_ == "Female" or sex_ == "female":
                        sex = 1
                    else:
                        print("Error in reading sex!", sex_)

                if l.startswith("# Dx:"):
                    arrs = l.strip().split(" ")
                    for arr in arrs[2].split(","):
                        try:
                            class_index = class_names.index(arr.rstrip())
                            label[class_index] = 1.0
                        except:
                            pass
            if label.sum() < 1:
                continue

            data_signals.append(recording)
            data_names.append(name)
            data_ages.append(age)
            data_sexes.append(sex)
            data_labels.append(label)

        data_names = np.array(data_names)
        print("data_names", len(data_names))
        data_ages = np.array(data_ages, dtype="float32")
        data_sexes = np.array(data_sexes, dtype="float32")
        data_labels = np.array(data_labels, dtype="float32")
        data_domains = deep_utils.find_domains(data_names)
        print("data_domains", len(data_domains))

    ################################

    test_names = np.load("test_val_names/test_names.npy")
    val_names = np.load("test_val_names/val_names.npy")
    test_inds = np.where(np.in1d(data_names, test_names))[0]
    val_inds = np.where(np.in1d(data_names, val_names))[0]
    train_inds = np.delete(
        np.arange(len(data_names)), np.concatenate([test_inds, val_inds])
    )

    ref_domains = list(set(data_domains))

    domain_masks = deep_utils.calc_domain_mask(data_domains, data_labels)

    data_domains = deep_utils.to_one_hot(data_domains, len(ref_domains))

    data_ages = data_ages[:, np.newaxis] / 100.0
    data_sexes = data_sexes[:, np.newaxis]

    test_size = len(test_inds)
    val_size = len(val_inds)
    train_size = len(train_inds)

    val_data = []
    for ind in val_inds:
        val_data.append(data_signals[ind])

    test_data = []
    for ind in test_inds:
        test_data.append(data_signals[ind])

    ##########################

    print("Elapsed:", time() - t0)
    print("Building model...")

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed + 3)

    AGE = tf.compat.v1.placeholder(tf.float32, [None, 1], name="AGE")
    SEX = tf.compat.v1.placeholder(tf.float32, [None, 1], name="SEX")

    X = tf.compat.v1.placeholder(tf.float32, [None, Length, 12], name="X")
    Y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="Y")
    Y_W = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="Y_W")
    D_M = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="D_M")
    Y_D = tf.compat.v1.placeholder(tf.float32, [None, len(ref_domains)], name="Y_D")

    lambda_d_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="lambda_d")
    
    score_ph = tf.compat.v1.placeholder(tf.float32, name="score")

    KEEP_PROB = tf.compat.v1.placeholder_with_default(1.0, (), "KEEP_PROB")

    # Pathway 1: CNN + LSTM
    e1 = tf.keras.layers.Conv1D(48, 9, strides=4, padding="valid", activation=tf.nn.relu, use_bias=True)(X)
    e2 = tf.keras.layers.Conv1D(64, 7, strides=3, activation=tf.nn.relu, use_bias=True)(e1)
    e3 = tf.keras.layers.Conv1D(80, 5, strides=2, activation=tf.nn.relu, use_bias=True)(e2)
    e4 = tf.keras.layers.Conv1D(96, 3, strides=2, activation=tf.nn.relu, use_bias=True)(e3)
    e5_0 = tf.keras.layers.MaxPooling1D(2, strides=2)(tf.keras.layers.Conv1D(112, 3, strides=2, activation=tf.nn.relu, use_bias=True)(e4))

    print(e5_0.shape)

    emb1 = keras.layers.Bidirectional(keras.layers.LSTM(128, recurrent_dropout=0.5, return_sequences=False))(e5_0)

    # # Pathway 2: CNN + Transformer
    class TransformerEncoderBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerEncoderBlock, self).__init__()
            self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation="relu"), 
                tf.keras.layers.Dense(embed_dim),
            ])
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, inputs, training=False):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    class PositionalEncoding(tf.keras.layers.Layer):
        def __init__(self, position, d_model):
            super(PositionalEncoding, self).__init__()
            self.pos_encoding = self.positional_encoding(position, d_model)

        def get_angles(self, position, i, d_model):
            angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return position * angles

        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(
                position=np.arange(position)[:, np.newaxis],
                i=np.arange(d_model)[np.newaxis, :],
                d_model=d_model
            )
            # Apply sin to even indices in the array; 2i
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            # Apply cos to odd indices in the array; 2i+1
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            
            pos_encoding = angle_rads[np.newaxis, ...]
            return tf.cast(pos_encoding, dtype=tf.float32)

        def call(self, inputs):
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


    e1 = tf.keras.layers.Conv1D(48, 19, strides=4, padding="valid", activation=tf.nn.relu, use_bias=True)(X)
    e2 = tf.keras.layers.MaxPooling1D(2, strides=2)(e1)
    seq_length = e2.shape[1]  # Use this for static shapes; for dynamic shapes, you might need a different approach
    d_model = e2.shape[2]

    pos_encoding_layer = PositionalEncoding(position=seq_length, d_model=d_model)
    e2_pos_encoded = pos_encoding_layer(e2)
    transformer_block = TransformerEncoderBlock(embed_dim=48, num_heads=4, ff_dim=128)
    transformer1_output = transformer_block(e2_pos_encoded)
    transformer_output = transformer_block(transformer1_output)
    emb2 = tf.reduce_mean(transformer_output, axis=1)  # Apply average pooling over the sequence dimension

    emb = tf.concat([emb1, emb2, AGE, SEX], axis=1)

    logs = []
    preds = []
    for i in range(num_classes):
        e6 = tf.nn.dropout(
            tf.compat.v1.layers.dense(emb, 100, activation=tf.nn.relu),
            rate=1 - (KEEP_PROB),
        )
        log_ = tf.compat.v1.layers.dense(e6, 1)
        pred_ = tf.nn.sigmoid(log_)
        logs.append(log_)
        preds.append(pred_)

    logits = tf.concat(logs, 1, name="out")
    
    print("Shape of emb:", emb.shape)
    print("Shape of logits:", logits.shape)
    pred = tf.concat(preds, 1, name="out_a")

    # gradient reversal for Domain Generalization:
    e5_p = tf.stop_gradient((1.0 + lambdaa_DG) * emb)
    e5_n = -lambdaa_DG * emb
    e5_d = e5_p + e5_n
    e6_d = tf.nn.dropout(
        tf.compat.v1.layers.dense(e5_d, 64, activation=tf.nn.relu), rate=1 - (KEEP_PROB)
    )
    logits_d = tf.compat.v1.layers.dense(e6_d, len(ref_domains))

    tr_vars = tf.compat.v1.trainable_variables()
    trainable_params = 0
    for i in range(len(tr_vars)):
        tmp = 1
        for j in tr_vars[i].get_shape().as_list():
            tmp *= j
        trainable_params += tmp
    print("# trainable parameters: ", trainable_params)

    MASK = tf.multiply(Y_W, D_M)
    
    print("logits shape:", logits.shape)
    print("logits_d shape:", logits_d.shape)
    print("length of ref_domains:", len(ref_domains))
    print("Y_W shape:", Y_W.shape)
    print("D_M shape:", D_M.shape)
    print("MASK shape:", MASK.shape)


    Loss = tf.reduce_mean(
        tf.multiply(
            MASK, tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
        )
    )
    
    Loss_d = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_d, labels=Y_D)
    )
    
    # Logging
    loss_summary = scalar('Loss', Loss)
    loss_d_summary = scalar('Domain_Loss', Loss_d)
    score_summary = scalar('Test_Score', score_ph)
    
    merged_summary = main_summary = tf.compat.v1.summary.merge([loss_summary, loss_d_summary])

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_opt = opt.minimize(Loss + lambdaa_d * Loss_d)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver()

    log_path = "logs/" + output_directory
    model_path = output_directory

    if not os.path.exists(log_path):
        os.makedirs(log_path)


    print("Training model...")
    
    def adjust_lambda(epoch, start_epoch=0, base_lambda=lambdaa_d, max_lambda=1.0, ramp_up_epochs=50):
        if epoch < start_epoch:
            return base_lambda
        elif epoch < start_epoch + ramp_up_epochs:
            return base_lambda + (max_lambda - base_lambda) * ((epoch - start_epoch) / ramp_up_epochs)
        return max_lambda

    def evaluate(data_x, data_y, data_m, data_a, data_s):
        loss_ce = []
        out = []

        n_b = len(data_y) // batch_size

        for bb in range(n_b):

            batch_data0 = []
            for bbbb in range(bb * batch_size, (bb + 1) * batch_size):
                batch_data0.append(data_x[bbbb])

            batch_data = deep_utils.prepare_data(
                batch_data0, Length, mod=np.random.randint(0, 2), aug=False
            )

            batch_label = data_y[bb * batch_size : (bb + 1) * batch_size]
            batch_weight = deep_utils.calc_weights(batch_label, class_weights)
            batch_mask = data_m[bb * batch_size : (bb + 1) * batch_size]
            batch_age = data_a[bb * batch_size : (bb + 1) * batch_size]
            batch_sex = data_s[bb * batch_size : (bb + 1) * batch_size]
            loss_ce_, out_ = sess.run(
                [Loss, "out_a:0"],
                feed_dict={
                    X: batch_data,
                    Y: batch_label,
                    Y_W: batch_weight,
                    D_M: batch_mask,
                    AGE: batch_age,
                    SEX: batch_sex,
                },
            )

            out.extend(out_)
            loss_ce.append(loss_ce_)

        out = np.array(out)

        return out, np.mean(loss_ce)

    epoch = 0
    step = 0
    # number of batches in train, validation, test data
    n_b = train_size // batch_size
    # n_b_v = val_size // batch_size
    # n_b_t = test_size // batch_size

    saver = tf.compat.v1.train.Saver()

    max_score = 0.1
    max_score_t = 0.1

    threshold = 0.1 * np.ones((num_classes))

    scores = []
    
    # mark time
    t0 = time()

    def random_crop(x, l_min):

        data = np.zeros_like(x)
        for i in range(x.shape[0]):
            cur_l = np.random.randint(l_min, x.shape[1])
            st = np.random.randint(0, x.shape[1] - cur_l)
            data[i, st : st + cur_l] = x[i, st : st + cur_l].copy()

        return data

    # Training loop
    while epoch < epoch_size:
        print("Epoch:", epoch)
        print("Elapsed:", time() - t0)
        batch_inds = np.random.permutation(train_size)
        for b in range(n_b):
            batch_ind = batch_inds[b * batch_size : (b + 1) * batch_size]

            # Prepare training batch
            train_batch = deep_utils.prepare_data(
                [data_signals[train_inds[bb]] for bb in batch_ind], Length, mod=np.random.randint(0, 2)
            )
    
            # Randomly crop the batch
            if np.random.rand() < 0.2:
                train_batch = random_crop(train_batch.copy(), Length // 2)

            # Prepare batch mask
            batch_mask = np.ones_like(domain_masks[train_inds[batch_ind]].copy()) if np.random.rand() < 0.5 else domain_masks[train_inds[batch_ind]].copy()

            # Prepare feed dictionary
            feed = {
                X: train_batch,
                Y: data_labels[train_inds[batch_ind]],
                Y_W: deep_utils.calc_weights(data_labels[train_inds[batch_ind]], class_weights),
                KEEP_PROB: 0.5,
                D_M: batch_mask,
                AGE: data_ages[train_inds[batch_ind]],
                SEX: data_sexes[train_inds[batch_ind]],
                Y_D: data_domains[train_inds[batch_ind]]
            }

            # Run training step
            sess.run(train_opt, feed)
            
            summary = sess.run(merged_summary, feed_dict=feed)
            # Add the summary to the TensorBoard
            summary_writer.add_summary(summary, step)

            step += 1

            # Validation and threshold adjustment
            if epoch >= 20 and step % n_step == 0:
                out, loss_ce = evaluate(val_data, data_labels[val_inds].copy(), domain_masks[val_inds].copy(), data_ages[val_inds].copy(), data_sexes[val_inds].copy())
                labels_ = (out > threshold).astype(int)
                score_v = evaluate_12ECG_score.compute_challenge_metric(class_weights, data_labels[val_inds][: len(out)].copy().astype(bool), labels_.astype(bool), list(class_names), "426783006")

                if score_v >= max_score or step % (1 * n_step) == 0:
                    out_th = np.concatenate([evaluate(val_data, data_labels[val_inds].copy(), domain_masks[val_inds].copy(), data_ages[val_inds].copy(), data_sexes[val_inds].copy())[0] for _ in range(2)], 0)
                    lbl_th = np.concatenate([data_labels[val_inds][: len(out_th1)].copy() for out_th1 in [out_th[:len(out_th)//2], out_th[len(out_th)//2:]]], 0)
                    threshold_ = threshold.copy()

                    for _ in range(2):
                        for jj in range(num_classes):
                            ss = 0.01
                            for th in range(1, 60, 2):
                                threshold[jj] = th / 100
                                labels_ = (out_th > threshold).astype(int)
                                score = evaluate_12ECG_score.compute_challenge_metric(class_weights, lbl_th.astype(bool), labels_.astype(bool), list(class_names), "426783006")
                                if score > ss:
                                    ss = score
                                    threshold_[jj] = th / 100
                    threshold = threshold_.copy()

                    # Evaluation on test data
                    out_th, _ = evaluate(test_data, data_labels[test_inds].copy(), domain_masks[test_inds].copy(), data_ages[test_inds].copy(), data_sexes[test_inds].copy())
                    lbl_th = data_labels[test_inds][: len(out_th)].copy()
                    labels_ = (out_th > threshold).astype(int)
                    score = evaluate_12ECG_score.compute_challenge_metric(class_weights, lbl_th.astype(bool), labels_.astype(bool), list(class_names), "426783006")
                    if score > max_score_t:
                        summary = sess.run(score_summary, feed_dict={score_ph: score})
                        summary_writer.add_summary(summary, epoch)
                        print("score:", score)
                        max_score_t = score
                        name = "epoch_" + str(epoch) + "_score_" + str(int(100 * score))
                        saver.save(sess, save_path=model_path + "/" + name)
                        np.save(model_path + "/threshold", threshold)
                    scores.append(score)
                    print("epoch:", epoch, "error_ce:", loss_ce, "score_v:", score_v, "score_t:", score)

        np.save(log_path + "/scores", scores)
        epoch += 1
    print(max_score_t)
