import os
import numpy as np
import csv
import tensorflow as tf
from tensorflow.compat.v1.summary import merge_all, scalar, FileWriter
from scipy.signal import find_peaks, filtfilt, butter


# from matplotlib import pyplot as plt
import deep_utils
import evaluate_12ECG_score
from time import time
from tensorflow import keras
from tqdm import tqdm
import concurrent.futures
import heartpy as hp


def train_12ECG_classifier_resnet(input_directory, output_directory):

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
    data_signals, data_names, data_ages, data_sexes, data_labels, hr_mins, hr_maxs, rmssds = deep_utils.load_pickles("training/processed")
    print("Data loaded successfully")
    print("hr_mins", hr_mins)
    print("hr_maxs", hr_maxs)
    data_names = np.array(data_names)
    data_ages = np.array(data_ages, dtype="float32")
    data_sexes = np.array(data_sexes, dtype="float32")
    data_labels = np.array(data_labels, dtype="float32")
    hr_mins = np.array(hr_mins, dtype="float32")
    hr_maxs = np.array(hr_maxs, dtype="float32")
    rmssds = np.array(rmssds, dtype="float32")
    # beats = np.array(beats_list, dtype="float32")
    # print if hr_mins or hr_maxs or rmssds have nan's
    print("hr_mins nan:", np.isnan(hr_mins).sum())
    print("hr_maxs nan:", np.isnan(hr_maxs).sum())
    print("rmssds nan:", np.isnan(rmssds).sum())
    
    
    print("hr_mins", hr_mins[0].shape)
    print("hr_maxs", hr_maxs[0].shape)

    data_domains = deep_utils.find_domains(data_names)

    data_domains = np.array(data_domains)
    
    print("data_domains", len(data_domains))
    ################################

    test_names = np.load("test_val_names/test_names.npy")
    val_names = np.load("test_val_names/val_names.npy")
    test_inds = np.where(data_domains == 4)[0]
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
    
    def squeeze_excitation_block(input_tensor, ratio=16):
        """Squeeze-and-Excitation Block."""
        channel_axis = -1
        filters = input_tensor.shape[channel_axis]
        se_shape = (1, filters)

        se = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
        se = tf.keras.layers.Reshape(se_shape)(se)
        se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
        se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
        
        return tf.keras.layers.multiply([input_tensor, se])

    def resnet_block(input_tensor, filters, kernel_size, use_se=True, strides=1):
        """A basic ResNet block with optional Squeeze-and-Excitation."""
        x = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if use_se:
            x = squeeze_excitation_block(x)

        shortcut = tf.keras.layers.Conv1D(filters, 1, strides=strides, padding='same')(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

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
    
    HR_min = tf.compat.v1.placeholder(tf.float32, [None, 1], name="HR_min")
    HR_max = tf.compat.v1.placeholder(tf.float32, [None, 1], name="HR_max")
    RMSSD = tf.compat.v1.placeholder(tf.float32, [None, 1], name="RMSSD")
    
    # Beats = tf.compat.v1.placeholder(tf.float32, [None, 3, 400], name="Beats")
    
    score_ph = tf.compat.v1.placeholder(tf.float32, name="score")

    KEEP_PROB = tf.compat.v1.placeholder_with_default(1.0, (), "KEEP_PROB")

    # Initial Convolution
    x = keras.layers.Conv1D(32, 16, strides=1, padding='same')(X)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # Residual Blocks
    filters = 64
    x = resnet_block(x, filters=filters, kernel_size=16, use_se=True, strides=2)
    filters *= 2
    x = resnet_block(x, filters=filters, kernel_size=8, use_se=True, strides=2)
    filters *= 2
    x = resnet_block(x, filters=filters, kernel_size=8, use_se=True, strides=2)
    x = resnet_block(x, filters=filters, kernel_size=5, use_se=True, strides=1)

    # Flatten and Concatenate with Handcrafted Features
    x = keras.layers.GlobalAveragePooling1D()(x)
    handcrafted_features = tf.concat([AGE, SEX, HR_min, HR_max, RMSSD], 1)
    combined_features = keras.layers.concatenate([x, handcrafted_features])

    # Final Dense Layer for Classification
    x = keras.layers.Dense(128, activation='relu')(combined_features)
    
    # Dropout and final dense layer
    x = tf.nn.dropout(
        tf.compat.v1.layers.dense(x, 100, activation=tf.nn.relu),
        rate=1 - (KEEP_PROB),
    )
    
    logits = tf.compat.v1.layers.dense(x, num_classes, name="out")
    pred = tf.nn.sigmoid(logits, name="out_a")
    # gradient reversal for Domain Generalization:
    e5_p = tf.stop_gradient((1.0 + lambdaa_DG) * x)
    e5_n = -lambdaa_DG * x
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

    def evaluate(data_x, data_y, data_m, data_a, data_s, data_hr_min, data_hr_max, data_rmssd):
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
            batch_hr_min = data_hr_min[bb * batch_size : (bb + 1) * batch_size]
            batch_hr_max = data_hr_max[bb * batch_size : (bb + 1) * batch_size]
            batch_rmssd = data_rmssd[bb * batch_size : (bb + 1) * batch_size]
            
            loss_ce_, out_ = sess.run(
                [Loss, "out_a:0"],
                feed_dict={
                    X: batch_data,
                    Y: batch_label,
                    Y_W: batch_weight,
                    D_M: batch_mask,
                    AGE: batch_age,
                    SEX: batch_sex,
                    HR_min: batch_hr_min,
                    HR_max: batch_hr_max,
                    RMSSD: batch_rmssd
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
            
            # Prepare heart rate data
            hr_mins = hr_mins.reshape(-1, 1)
            hr_maxs = hr_maxs.reshape(-1, 1)
            rmssds = rmssds.reshape(-1, 1)      
            
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
                HR_min: hr_mins[train_inds[batch_ind]],
                HR_max: hr_maxs[train_inds[batch_ind]],
                RMSSD: rmssds[train_inds[batch_ind]]
            }

            # Run training step
            sess.run(train_opt, feed)
            
            summary = sess.run(merged_summary, feed_dict=feed)
            # Add the summary to the TensorBoard
            summary_writer.add_summary(summary, step)

            step += 1

            # Validation and threshold adjustment
            if epoch >= 10 and step % n_step == 0:
                out, loss_ce = evaluate(val_data, data_labels[val_inds].copy(), domain_masks[val_inds].copy(), data_ages[val_inds].copy(), data_sexes[val_inds].copy(), hr_mins[val_inds].copy(), hr_maxs[val_inds].copy(), rmssds[val_inds].copy())
                labels_ = (out > threshold).astype(int)
                score_v = evaluate_12ECG_score.compute_challenge_metric(class_weights, data_labels[val_inds][: len(out)].copy().astype(bool), labels_.astype(bool), list(class_names), "426783006")
                if score_v >= max_score or step % (1 * n_step) == 0:
                    out_th = np.concatenate([evaluate(val_data, data_labels[val_inds].copy(), domain_masks[val_inds].copy(), data_ages[val_inds].copy(), data_sexes[val_inds].copy(), hr_mins[val_inds].copy(), hr_maxs[val_inds].copy(), rmssds[val_inds].copy())[0] for _ in range(2)], 0)
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
                    out_th, _ = evaluate(test_data, data_labels[test_inds].copy(), domain_masks[test_inds].copy(), data_ages[test_inds].copy(), data_sexes[test_inds].copy(), hr_mins[test_inds].copy(), hr_maxs[test_inds].copy(), rmssds[test_inds].copy())
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
