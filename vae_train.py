import os
import sys
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
import pandas as pd
import matplotlib.pyplot as plt



def train_12ECG_classifier_vae(input_directory, output_directory):

    tf.compat.v1.disable_eager_execution()

    logdir = "tensorboard/" + output_directory
    summary_writer = FileWriter(logdir)

    seed = 0
    np.random.seed(seed)

    epoch_size = 100

    learning_rate = 8e-4
    Length = 2800

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
    
    score_ph = tf.compat.v1.placeholder(tf.float32, name="score")

    KEEP_PROB = tf.compat.v1.placeholder_with_default(1.0, (), "KEEP_PROB")
        
    # Define layers
    conv1 = tf.keras.layers.Conv1D(64, 13, strides=4b, padding="same", activation='relu')
    conv2 = tf.keras.layers.Conv1D(128, 9, strides=4, padding="same", activation='relu')
    conv3 = tf.keras.layers.Conv1D(256, 5, strides=4, padding="same", activation='relu')
    max_pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
    lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))

    def create_encoder(X):
        e1 = conv1(X)
        e2 = conv2(e1)
        e3 = conv3(e2)
        e4 = max_pool(e3)
        emb = lstm_layer(e3)
        return emb
    
    emb = create_encoder(X)
        
    def create_decoder(emb):
        h1 = tf.keras.layers.Dense(units=175 * 32, activation='relu')(emb)
        h1 = tf.keras.layers.Reshape((175, 32))(h1)
        h2 = tf.keras.layers.Conv1DTranspose(128, 3, strides=4, padding='same', activation='relu')(h1)
        h3 = tf.keras.layers.Conv1DTranspose(64, 3, strides=4, padding='same', activation='relu')(h2)
        decoded_output = tf.keras.layers.Conv1D(12, 3, padding='same', activation='tanh')(h3)
        return decoded_output

    # Usage
    decoded = create_decoder(emb)

    emb = tf.concat([emb, AGE, SEX], axis=1)

    logs = []
    preds = []
    for i in range(num_classes):
        e6 = tf.keras.layers.Dropout(rate=1 - (KEEP_PROB))(tf.keras.layers.Dense(100, activation='relu')(emb))
        log_ = tf.keras.layers.Dense(1)(e6)
        pred_ = tf.keras.layers.Activation('sigmoid')(log_)
        logs.append(log_)
        preds.append(pred_)

    logits = tf.concat(logs, 1, name="out")

    # Display the shapes for verification
    print("Shape of emb1 (used for prediction):", emb.shape)
    print("Shape of logits:", logits.shape)
    pred = tf.concat(preds, 1, name="out_a")
    
    # # gradient reversal for Domain Generalization:
    # e5_p = tf.stop_gradient((1.0 + 1e-3) * emb)
    # e5_n = -1e-3 * emb
    # e5_d = e5_p + e5_n
    # e6_d = tf.nn.dropout(
    #     tf.compat.v1.layers.dense(e5_d, 64, activation=tf.nn.relu), rate=1 - (KEEP_PROB)
    # )
    # logits_d = tf.compat.v1.layers.dense(e6_d, len(ref_domains))

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
    print("length of ref_domains:", len(ref_domains))
    print("Y_W shape:", Y_W.shape)
    print("D_M shape:", D_M.shape)
    print("MASK shape:", MASK.shape)

    Loss = tf.reduce_mean(
        tf.multiply(
            MASK, tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
        )
    )
    
    # Loss_d = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(logits=logits_d, labels=Y_D)
    # )
    
    # Reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.square(X - decoded))    
    # Logging
    loss_summary = scalar('Loss', Loss)
    reconstruction_loss_summary = scalar('Reconstruction_Loss', reconstruction_loss)
    # loss_d_summary = scalar('Domain_Loss', Loss_d)
    score_summary = scalar('Test_Score', score_ph)
    
    merged_summary = main_summary = tf.compat.v1.summary.merge([loss_summary, reconstruction_loss_summary])

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    Total_loss = Loss + reconstruction_loss

    train_opt = opt.minimize(Total_loss)

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
    
    def one_cycle_lr_schedule(step, total_steps, lr_max, lr_start=0.0):
        # The step at which to reach the maximum learning rate
        rampup_steps = total_steps // 2
        # The step at which to reduce the learning rate to zero
        rampdown_steps = total_steps // 2
    
        if step < rampup_steps:
            lr = ((lr_max - lr_start) / rampup_steps) * step + lr_start
        elif step < rampup_steps + rampdown_steps:
            lr = ((lr_start - lr_max) / rampdown_steps) * (step - rampup_steps) + lr_max
        else:
            lr = lr_start
    
        return lr

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
    
    def adjust_lambda(epoch, start_epoch=5, base_lambda=0.0, max_lambda=2e-3, ramp_up_epochs=30):
        if epoch < start_epoch:
            return base_lambda
        elif epoch < start_epoch + ramp_up_epochs:
            return base_lambda + (max_lambda - base_lambda) * ((epoch - start_epoch) / ramp_up_epochs)
        return max_lambda


    # Training loop
    while epoch < epoch_size:
        print("Epoch:", epoch)
        lr = one_cycle_lr_schedule(epoch, epoch_size, 0.001)
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        train_opt = opt.minimize(Total_loss)
        batch_inds = np.random.permutation(train_size)
        for b in range(n_b):
            batch_ind = batch_inds[b * batch_size : (b + 1) * batch_size]

            
            # Prepare training batch
            train_batch = deep_utils.prepare_data(
                [data_signals[train_inds[bb]] for bb in batch_ind], Length, mod=np.random.randint(0, 2)
            )

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
            
            decoded_values = sess.run(decoded, feed_dict=feed)
            
            # Log/plot one signal and its reconstruction per epoch
            if b == 0:
                plt.figure(figsize=(10, 5))
                plt.plot(train_batch[0, :, 0], label='Original')
                plt.plot(decoded_values[0, :, 0], label='Reconstruction')
                plt.legend()
                plt.show()
            
            
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
