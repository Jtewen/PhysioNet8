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
from collections import Counter



def train_12ECG_classifier_multilevel(input_directory, output_directory):

    tf.compat.v1.disable_eager_execution()

    logdir = "tensorboard/" + output_directory
    summary_writer = FileWriter(logdir)

    seed = 0
    np.random.seed(seed)

    epoch_size = 100

    learning_rate = 8e-4
    Length = 5600

    lambdaa_d = 1e-1

    lambdaa_DG = 2e-3

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
    print("Loading data from files...")
    data_signals, data_names, data_ages, data_sexes, data_labels = deep_utils.load_pickles("training/processed")
    print("Data loaded successfully")
    data_names = np.array(data_names)
    data_ages = np.array(data_ages, dtype="float32")
    data_sexes = np.array(data_sexes, dtype="float32")
    data_labels = np.array(data_labels, dtype="float32")


    data_domains = deep_utils.find_domains(data_names)

    data_domains = np.array(data_domains)
    
    print("data_domains", len(data_domains))
    
    ################################

    test_names = np.load("test_val_names/test_names.npy")
    val_names = np.load("test_val_names/val_names.npy")
    test_inds = np.where(np.in1d(data_names, test_names))[0]
    val_inds = np.where(np.in1d(data_names, val_names))[0]
    train_inds = np.delete(
        np.arange(len(data_names)), np.concatenate([test_inds, val_inds])
    )
    
    domain_counts = Counter(data_domains)
    print(domain_counts)
    ref_domains = list(set(data_domains))
    print("ref_domains", ref_domains)
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
    
    def concentration_layer(input_tensor, filters=64, dropout_rate=0.3, pool_size=2):
        x = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(input_tensor)
        x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
        return x
    
    e0 = tf.keras.layers.Conv1D(48, 12, strides=4, padding="valid", activation='relu', use_bias=True)(X)

    # First block
    e1 = tf.keras.layers.Conv1D(48, 9, strides=3, padding="valid", activation='relu', use_bias=True)(e0)
    e2 = tf.keras.layers.Conv1D(48, 7, strides=1, padding="valid", activation='relu', use_bias=True)(e1)
    e3 = tf.keras.layers.Conv1D(48, 5, strides=1, padding="valid", activation='relu', use_bias=True)(e2)
    block1 = concentration_layer(e3)
    e3 = tf.keras.layers.BatchNormalization()(e3)
    
    # Second block
    e4 = tf.keras.layers.Conv1D(64, 5, strides=3, padding="valid", activation='relu', use_bias=True)(e3)
    e5 = tf.keras.layers.Conv1D(64, 3, strides=1, padding="valid", activation='relu', use_bias=True)(e4)
    e6 = tf.keras.layers.Conv1D(64, 3, strides=1, padding="valid", activation='relu', use_bias=True)(e5)
    block2 = concentration_layer(e6)
    e6 = tf.keras.layers.BatchNormalization()(e6)
    
    # Third block
    e7 = tf.keras.layers.Conv1D(80, 3, strides=2, padding="valid", activation='relu', use_bias=True)(e6)
    e8 = tf.keras.layers.Conv1D(80, 3, strides=1, padding="valid", activation='relu', use_bias=True)(e7)
    e9 = tf.keras.layers.Conv1D(80, 3, strides=1, padding="valid", activation='relu', use_bias=True)(e8)
    block3 = concentration_layer(e9)
    
    features = tf.keras.layers.concatenate([block1, block2, block3], axis=1)
    bidirectional_gru_features = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=False))(features)
    emb = tf.concat([bidirectional_gru_features, AGE, SEX], axis=1)

    logs = []
    preds = []
    for i in range(num_classes):
        e6 = tf.keras.layers.Dropout(rate=1 - KEEP_PROB)(tf.keras.layers.Dense(100, activation='relu')(emb))
        log_ = tf.keras.layers.Dense(1)(e6)
        pred_ = tf.keras.layers.Activation('sigmoid')(log_)
        logs.append(log_)
        preds.append(pred_)
    
    logits = tf.keras.layers.Concatenate(axis=1, name="out")(logs)
    
    print("Shape of emb:", emb.shape)
    print("Shape of logits:", logits.shape)
    pred = tf.keras.layers.Concatenate(axis=1, name="out_a")(preds)
    
    # gradient reversal for Domain Generalization:
    e5_p = tf.stop_gradient((1.0 + lambdaa_DG) * emb)
    e5_n = -lambdaa_DG * emb
    e5_d = e5_p + e5_n
    e6_d = tf.keras.layers.Dropout(rate=1 - KEEP_PROB)(tf.keras.layers.Dense(64, activation='relu')(e5_d))
    logits_d = tf.keras.layers.Dense(len(ref_domains))(e6_d)
    
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
                KEEP_PROB: 0.7,
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
