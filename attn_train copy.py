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
from keras_self_attention import SeqSelfAttention



def train_12ECG_classifier(input_directory, output_directory):

    tf.compat.v1.disable_eager_execution()

    seed = 0
    np.random.seed(seed)

    epoch_size = 100

    learning_rate = 8e-4
    Length = 5600

    lambdaa_d = 1e-1

    lambdaa_DG = 1e-3

    batch_size = 128

    n_step = 80

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

        header_files = []
        for f in os.listdir(input_directory):
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
        keras.layers.LSTM(128, recurrent_dropout=0.5, return_sequences=False)
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
        keras.layers.LSTM(128, recurrent_dropout=0.5, return_sequences=False)
    )(e5_0)

    lstm_emb = tf.concat([emb1, emb2, AGE, SEX], axis=1)
    
    attention_layer = SeqSelfAttention(attention_activation='sigmoid')(lstm_emb)

    emb = tf.concat([emb, attention_layer], axis=1)


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

    while epoch < epoch_size:
        batch_inds = np.random.permutation(train_size)
        for b in range(n_b):
            batch_ind = batch_inds[b * batch_size : (b + 1) * batch_size]

            train_batch0 = []
            for bb in batch_ind:
                train_batch0.append(data_signals[train_inds[bb]])

            train_batch = deep_utils.prepare_data(
                train_batch0, Length, mod=np.random.randint(0, 2)
            )

            if np.random.rand() < 0.2:
                train_batch = random_crop(train_batch.copy(), Length // 2)

            batch_mask = domain_masks[train_inds[batch_ind]].copy()

            #            if np.random.rand() < 0.1:
            #                scale = np.random.rand()
            #                batch_mask = scale * batch_mask + (1 - scale)

            if np.random.rand() < 0.5:
                batch_mask = np.ones_like(batch_mask)

            batch_age = data_ages[train_inds[batch_ind]]
            batch_sex = data_sexes[train_inds[batch_ind]]
            batch_domain = data_domains[train_inds[batch_ind]]

            batch_label = data_labels[train_inds[batch_ind]]
            batch_weight = deep_utils.calc_weights(batch_label, class_weights)
            feed = {
                X: train_batch,
                Y: batch_label,
                Y_W: batch_weight,
                KEEP_PROB: 0.5,
                D_M: batch_mask,
                AGE: batch_age,
                SEX: batch_sex,
                Y_D: batch_domain,
            }
            sess.run(train_opt, feed)

            step += 1

            if epoch < 20:
                continue

            ################# validation  ######################
            if step % n_step == 0:

                out, loss_ce = evaluate(
                    val_data,
                    data_labels[val_inds].copy(),
                    domain_masks[val_inds].copy(),
                    data_ages[val_inds].copy(),
                    data_sexes[val_inds].copy(),
                )

                labels_ = out > threshold
                labels_ = labels_.astype(int)

                score_v = evaluate_12ECG_score.compute_challenge_metric(
                    class_weights,
                    data_labels[val_inds][: len(out)].copy().astype(bool),
                    labels_.astype(bool),
                    list(class_names),
                    "426783006",
                )

                if score_v >= max_score or step % (1 * n_step) == 0:

                    ####################### find thresholds from validation data ######################

                    out_th1, _ = evaluate(
                        val_data,
                        data_labels[val_inds].copy(),
                        domain_masks[val_inds].copy(),
                        data_ages[val_inds].copy(),
                        data_sexes[val_inds].copy(),
                    )

                    out_th2, _ = evaluate(
                        val_data,
                        data_labels[val_inds].copy(),
                        domain_masks[val_inds].copy(),
                        data_ages[val_inds].copy(),
                        data_sexes[val_inds].copy(),
                    )

                    out_th = np.concatenate([out_th1, out_th2], 0)
                    lbl_th = np.concatenate(
                        [
                            data_labels[val_inds][: len(out_th1)].copy(),
                            data_labels[val_inds][: len(out_th2)].copy(),
                        ],
                        0,
                    )

                    out_ = out_th.copy()

                    threshold_ = threshold.copy()

                    for _ in range(2):

                        for jj in range(num_classes):
                            threshold = threshold_.copy()
                            out_ = out_th.copy()
                            ss = 0.01
                            for th in range(1, 60, 2):
                                threshold[jj] = th / 100
                                labels_ = out_ > threshold
                                labels_ = labels_.astype(int)

                                score = evaluate_12ECG_score.compute_challenge_metric(
                                    class_weights,
                                    lbl_th.astype(bool),
                                    labels_.astype(bool),
                                    list(class_names),
                                    "426783006",
                                )

                                if score > ss:
                                    ss = score
                                    threshold_[jj] = th / 100

                    threshold = threshold_.copy()

                    ####################### evaluation on test data ######################

                    out_th, _ = evaluate(
                        test_data,
                        data_labels[test_inds].copy(),
                        domain_masks[test_inds].copy(),
                        data_ages[test_inds].copy(),
                        data_sexes[test_inds].copy(),
                    )

                    lbl_th = data_labels[test_inds][: len(out_th)].copy()

                    labels_ = out_th > threshold
                    labels_ = labels_.astype(int)

                    score = evaluate_12ECG_score.compute_challenge_metric(
                        class_weights,
                        lbl_th.astype(bool),
                        labels_.astype(bool),
                        list(class_names),
                        "426783006",
                    )

                    if score > max_score_t:
                        print("score:", score)
                        max_score_t = score

                        name = "epoch_" + str(epoch) + "_score_" + str(int(100 * score))
                        saver.save(sess, save_path=model_path + "/" + name)
                        np.save(model_path + "/threshold", threshold)

                    scores.append(score)

                    print(
                        "epoch:",
                        epoch,
                        "error_ce:",
                        loss_ce,
                        "score_v:",
                        score_v,
                        "score_t:",
                        score,
                    )

        np.save(log_path + "/scores", scores)

        epoch += 1
    print(max_score_t)
