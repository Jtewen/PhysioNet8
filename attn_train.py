import os
import numpy as np
import csv
import tensorflow as tf

# from matplotlib import pyplot as plt
import deep_utils
import evaluate_12ECG_score
from time import time
from tensorflow import keras
from tqdm import tqdm
import glob


def train_12ECG_classifier_attn(input_directory, output_directory):

    tf.compat.v1.disable_eager_execution()

    seed, epoch_size, learning_rate, Length, lambdaa_d, lambdaa_DG, batch_size, n_step, N_data = 0, 100, 4e-3, 5600, 1e-1, 1e-3, 128, 80, 10000
    np.random.seed(seed)

    print("Loading data...")
    t0 = time()

    class_names = np.genfromtxt("weights.csv", delimiter=",", max_rows=1, dtype=str)[1:]
    num_classes = len(class_names)
    class_weights = np.genfromtxt("weights.csv", delimiter=",", skip_header=1, dtype=float)[:, 1:]

    os.makedirs(output_directory, exist_ok=True)
    np.save(f"{output_directory}/class_names.npy", class_names)

    header_files = glob.glob(f"{input_directory}/*.hea")
    num_files = len(header_files)
    print("num files", num_files)

    data_signals, data_names, data_ages, data_sexes, data_labels = [], [], [], [], []

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
                age = 50 if age_ in ["Nan", "NaN"] else int(age_)

            if l.startswith("# Sex:"):
                sex_ = l.strip().split(" ")[2].lower()
                sex = 0 if sex_ in ["male", "m"] else 1 if sex_ in ["female", "f"] else print("Error in reading sex!", sex_)

            if l.startswith("# Dx:"):
                arrs = l.strip().split(" ")[2].split(",")
                for arr in arrs:
                    if arr.rstrip() in class_names:
                        label[np.where(class_names == arr.rstrip())[0][0]] = 1.0
        if np.sum(label) < 1:
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

    test_names, val_names = np.load("test_val_names/test_names.npy"), np.load("test_val_names/val_names.npy")
    test_inds, val_inds = np.where(np.in1d(data_names, test_names))[0], np.where(np.in1d(data_names, val_names))[0]
    train_inds = np.delete(np.arange(len(data_names)), np.concatenate([test_inds, val_inds]))

    ref_domains = list(set(data_domains))
    domain_masks = deep_utils.calc_domain_mask(data_domains, data_labels)
    data_domains = deep_utils.to_one_hot(data_domains, len(ref_domains))

    data_ages = data_ages[:, np.newaxis] / 100.
    data_sexes = data_sexes[:, np.newaxis]

    test_size, val_size, train_size = len(test_inds), len(val_inds), len(train_inds)

    val_data = [data_signals[ind] for ind in val_inds]
    test_data = [data_signals[ind] for ind in test_inds]

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

    KEEP_PROB = tf.compat.v1.placeholder_with_default(1.0, (), "KEEP_PROB")

    e1 = tf.compat.v1.layers.conv1d(
        X, 48, 9, strides=4, padding="valid", activation=tf.nn.relu, use_bias=True
    )
    e2 = tf.compat.v1.layers.conv1d(
        e1, 64, 7, strides=3, activation=tf.nn.relu, use_bias=True
    )
    e3 = tf.compat.v1.layers.conv1d(
        e2, 80, 5, strides=2, activation=tf.nn.relu, use_bias=True
    )
    e4 = tf.compat.v1.layers.conv1d(
        e3, 96, 3, strides=2, activation=tf.nn.relu, use_bias=True
    )
    e5_0 = tf.compat.v1.layers.max_pooling1d(
        tf.compat.v1.layers.conv1d(
            e4, 112, 3, strides=2, activation=tf.nn.relu, use_bias=True
        ),
        2,
        2,
    )

    print(e5_0.shape)

    emb1 = keras.layers.Bidirectional(
        keras.layers.LSTM(128, recurrent_dropout=0.5, return_sequences=True)
    )(e5_0)

    e1 = tf.compat.v1.layers.conv1d(
        X, 48, 19, strides=4, padding="valid", activation=tf.nn.relu, use_bias=True
    )
    e2 = tf.compat.v1.layers.conv1d(
        e1, 64, 15, strides=3, activation=tf.nn.relu, use_bias=True
    )
    e3 = tf.compat.v1.layers.conv1d(
        e2, 80, 11, strides=2, activation=tf.nn.relu, use_bias=True
    )
    e4 = tf.compat.v1.layers.conv1d(
        e3, 96, 9, strides=2, activation=tf.nn.relu, use_bias=True
    )
    e5_0 = tf.compat.v1.layers.max_pooling1d(
        tf.compat.v1.layers.conv1d(
            e4, 112, 7, strides=2, activation=tf.nn.relu, use_bias=True
        ),
        2,
        2,
    )

    print(e5_0.shape)

    emb2 = keras.layers.Bidirectional(
        keras.layers.LSTM(128, recurrent_dropout=0.5, return_sequences=True)
    )(e5_0)
    
    def attention(inputs, name):
        # inputs should be in [batch_size, time_steps, input_dim]
        with tf.compat.v1.variable_scope(name):
            # Learn a set of weights which scores how important each timestep's input is
            hidden_size = inputs.shape[2]  # D value - hidden size of the RNN layer
            
            # Trainable parameters
            w_omega = tf.Variable(tf.compat.v1.random_normal([hidden_size, 1], stddev=0.1))
            b_omega = tf.Variable(tf.compat.v1.random_normal([1], stddev=0.1))
            u_omega = tf.Variable(tf.compat.v1.random_normal([1], stddev=0.1))
            
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            # the shape of v is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w_omega) + b_omega)
            
            # For each of the timestamps its vector of size A from 'v' is reduced with 'u' vector
            # After: Reshape u_omega to be 2D before the multiplication
            u_omega_reshaped = tf.reshape(u_omega, [-1, 1])  # Reshape u_omega to 2D
            vu = tf.matmul(v, u_omega_reshaped)  # Now both operands are 2D
            exps = tf.reshape(tf.exp(vu), [-1, inputs.shape[1]])  # B,T
            alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  # B,T
            
            # Output of Bi-RNN is reduced with attention vector; the result has (B,D) shape
            output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, inputs.shape[1], 1]), 1)
            
            return output, alphas

    # # Applying attention mechanism after LSTM layers
    print("Input shape to attention:", emb1.shape)
    emb1_attention, alphas1 = attention(emb1, name='attention1')
    emb2_attention, alphas2 = attention(emb2, name='attention2')

    # # Combining attention outputs
    emb = tf.concat([emb1_attention, emb2_attention], axis=1)
    print("Shape of emb:", emb.get_shape().as_list())
    # emb = tf.concat([emb1, emb2], axis=1)


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

    def random_crop(x, l_min):

        data = np.zeros_like(x)
        for i in range(x.shape[0]):
            cur_l = np.random.randint(l_min, x.shape[1])
            st = np.random.randint(0, x.shape[1] - cur_l)
            data[i, st : st + cur_l] = x[i, st : st + cur_l].copy()

        return data

    # Training loop
    while epoch < epoch_size:
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
                Y_D: data_domains[train_inds[batch_ind]],
            }

            # Run training step
            sess.run(train_opt, feed)
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
