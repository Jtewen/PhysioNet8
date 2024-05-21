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



def train_12ECG_classifier_resnet2(input_directory, output_directory):

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
    test_inds = domain_4_inds = np.where(data_domains == 4)[0]
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
    
    f1_t_ph = tf.compat.v1.placeholder(tf.float32, name="f1_t")
    fb_t_ph = tf.compat.v1.placeholder(tf.float32, name="fb_t")
    gb_t_ph = tf.compat.v1.placeholder(tf.float32, name="gb_t")
    
    f1_v_ph = tf.compat.v1.placeholder(tf.float32, name="f1_v")
    fb_v_ph = tf.compat.v1.placeholder(tf.float32, name="fb_v")
    gb_v_ph = tf.compat.v1.placeholder(tf.float32, name="gb_v")
    
    macro_auroc_t_ph = tf.compat.v1.placeholder(tf.float32, name="macro_auroc_t")
    micro_auprc_t_ph = tf.compat.v1.placeholder(tf.float32, name="micro_auprc_t")
    macro_auroc_v_ph = tf.compat.v1.placeholder(tf.float32, name="macro_auroc_v")
    micro_auprc_v_ph = tf.compat.v1.placeholder(tf.float32, name="micro_auprc_v")

    accuracy_t_ph = tf.compat.v1.placeholder(tf.float32, name="accuracy_t")
    accuracy_v_ph = tf.compat.v1.placeholder(tf.float32, name="accuracy_v")

    KEEP_PROB = tf.compat.v1.placeholder_with_default(1.0, (), "KEEP_PROB")
    
    lr = tf.compat.v1.placeholder(tf.float32, name="lr")
        
    def res_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name="res_block"):
        conv = tf.keras.layers.Conv1D(filters, kernel_size, strides=stride, padding="same", name=name + "_conv1")(x)
        conv = tf.keras.layers.BatchNormalization(name=name + "_bn1")(conv)
        conv = tf.keras.layers.Activation("relu", name=name + "_relu1")(conv)
        
        conv = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", name=name + "_conv2")(conv)
        conv = tf.keras.layers.BatchNormalization(name=name + "_bn2")(conv)
        
        if conv_shortcut:
            shortcut = tf.keras.layers.Conv1D(filters, 1, strides=stride, name=name + "_conv_shortcut")(x)
            shortcut = tf.keras.layers.BatchNormalization(name=name + "_bn_shortcut")(shortcut)
        else:
            shortcut = x
        
        output = tf.keras.layers.Add(name=name + "_add")([conv, shortcut])
        output = tf.keras.layers.Activation("relu", name=name + "_relu2")(output)
        
        return output

    def create_resnet18(x, name="resnet18"):
        conv0 = tf.keras.layers.Conv1D(64, 7, strides=2, padding="same", name=name + "_conv1")(x)
        conv0 = tf.keras.layers.BatchNormalization(name=name + "_bn1")(conv0)
        conv0 = tf.keras.layers.Activation("relu", name=name + "_relu1")(conv0)
        
        conv0 = tf.keras.layers.MaxPooling1D(3, strides=2, padding="same", name=name + "_maxpool1")(conv0)
        
        conv1 = res_block(conv0, 64, name=name + "_resblock1")
        conv2 = res_block(conv1, 64, name=name + "_resblock2")
        
        conv3 = res_block(conv2, 128, stride=2, conv_shortcut=True, name=name + "_resblock3")
        conv4 = res_block(conv3, 128, name=name + "_resblock4")
        
        conv5 = res_block(conv4, 256, stride=2, conv_shortcut=True, name=name + "_resblock5")
        conv6 = res_block(conv5, 256, name=name + "_resblock6")
        
        conv7 = res_block(conv6, 512, stride=2, conv_shortcut=True, name=name + "_resblock7")
        conv8 = res_block(conv7, 512, name=name + "_resblock8")
                        
        return [conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8]

    feature_maps = create_resnet18(X)

    def extraction_pipeline(x, filters, name="extraction"):
        x = tf.keras.layers.Conv1D(filters, 1, padding="same", name=name + "_conv")(x)
        x = tf.keras.layers.SpatialDropout1D(rate=0.5, name=name + "_dropout")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name=name + "_gap")(x)
        return x

    # Adjust and concatenate feature maps
    extracted_features = []
    for i, fmap in enumerate(feature_maps):
        adjusted_feature = extraction_pipeline(fmap, filters=64, name=f"extraction_{i}")
        extracted_features.append(adjusted_feature)

    # Concatenate all adjusted feature maps
    concatenated_features = tf.keras.layers.Concatenate()(extracted_features)

    
    emb = tf.concat([concatenated_features, AGE, SEX], axis=1)
    
    print("emb shape:", emb.shape)
    
    #GRL
    class GradientReversal(tf.keras.layers.Layer):
        def __init__(self, lambda_):
            super(GradientReversal, self).__init__()
            self.lambda_ = lambda_
    
        @tf.custom_gradient
        def call(self, x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad
    
    reversed_emb = GradientReversal(1e-3)(emb)

    # Primary task gate layer
    g_y = tf.keras.layers.Dense(emb.shape[-1], activation="sigmoid")(emb)
    gated_emb_y = tf.multiply(emb, g_y)
    d_y = tf.keras.layers.Dense(100, activation="relu")(gated_emb_y)
    dropout_a_y = tf.keras.layers.Dropout(0.5)(d_y)
    logits = tf.keras.layers.Dense(num_classes)(dropout_a_y)
    preds = tf.nn.sigmoid(logits, name="out_a")

    
    # Domain classification gate layer
    g_d = tf.keras.layers.Dense(emb.shape[-1], activation="sigmoid")(reversed_emb)
    gated_emb_d = tf.multiply(reversed_emb, g_d)
    d_d = tf.keras.layers.Dense(100, activation="relu")(gated_emb_d)
    dropout_d_d = tf.keras.layers.Dropout(0.5)(d_d)
    logits_d = tf.keras.layers.Dense(len(ref_domains))(dropout_d_d)
    preds_d = tf.nn.softmax(logits_d, name="out_d")

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
    
    Loss_d = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_d, labels=Y_D)
    )
    
    # Compute orthogonal loss on gate scores
    g_y_norm = tf.nn.l2_normalize(g_y, axis=-1)
    g_d_norm = tf.nn.l2_normalize(g_d, axis=-1)
    Loss_o = tf.reduce_mean(tf.square(tf.reduce_sum(tf.multiply(g_y_norm, g_d_norm), axis=-1)))

    # Logging
    loss_summary = scalar('Loss', Loss)
    loss_d_summary = scalar('Domain_Loss', Loss_d)
    loss_o_summary = scalar('Orthogonal_Loss', Loss_o)
    
    merged_summary = main_summary = tf.compat.v1.summary.merge([loss_summary, loss_d_summary, loss_o_summary])


    accuracy_t_summary = scalar('Test_Accuracy', accuracy_t_ph)
    f1_t_summary = scalar('Test_F1', f1_t_ph)
    fb_t_summary = scalar('Test_Fbeta', fb_t_ph)
    gb_t_summary = scalar('Test_Gbeta', gb_t_ph)
    accuracy_v_summary = scalar('Val_Accuracy', accuracy_v_ph)
    f1_v_summary = scalar('Val_F1', f1_v_ph)
    fb_v_summary = scalar('Val_Fbeta', fb_v_ph)
    gb_v_summary = scalar('Val_Gbeta', gb_v_ph)
    
    macro_auroc_t_summary = scalar('Test_Macro_AUROC', macro_auroc_t_ph)
    micro_auprc_t_summary = scalar('Test_Micro_AUPRC', micro_auprc_t_ph)
    macro_auroc_v_summary = scalar('Val_Macro_AUROC', macro_auroc_v_ph)
    micro_auprc_v_summary = scalar('Val_Micro_AUPRC', micro_auprc_v_ph)
    
    accuracy_t_summary = scalar('Test_Accuracy', accuracy_t_ph)
    accuracy_v_summary = scalar('Val_Accuracy', accuracy_v_ph)

    metric_summary = tf.compat.v1.summary.merge([accuracy_t_summary, f1_t_summary, fb_t_summary, gb_t_summary, accuracy_v_summary, f1_v_summary, fb_v_summary, gb_v_summary, macro_auroc_t_summary, micro_auprc_t_summary, macro_auroc_v_summary, micro_auprc_v_summary])    

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

    Total_loss = Loss + 0.1 * Loss_d + 0.05 * Loss_o

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
    
    def one_cycle_lr_schedule(step, total_steps, lr_max, lr_start=0.0004):
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

    max_fb = 0.01
    max_fb_t = 0.01

    scores = []

    # Training loop
    while epoch < epoch_size:
        print("Epoch:", epoch)
        lr_val = one_cycle_lr_schedule(epoch, epoch_size, 0.001)
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
                Y_D: data_domains[train_inds[batch_ind]],
                lr: lr_val
            }

            # Run training step
            sess.run(train_opt, feed)
            
            summary = sess.run(merged_summary, feed_dict=feed)
            # Add the summary to the TensorBoard
            summary_writer.add_summary(summary, step)

            step += 1

            # Validation
            if epoch >= 5 and step % n_step == 0:
                out, loss_ce = evaluate(val_data, data_labels[val_inds].copy(), domain_masks[val_inds].copy(), data_ages[val_inds].copy(), data_sexes[val_inds].copy())
                macro_auroc_v, micro_auprc_v, accuracy_v, f1_v, fb_v, gb_v = evaluate_12ECG_score.compute_metrics(data_labels[val_inds], out)
                if step % (1 * n_step) == 0:
                    # Evaluation on test data
                    out_th, _ = evaluate(test_data, data_labels[test_inds].copy(), domain_masks[test_inds].copy(), data_ages[test_inds].copy(), data_sexes[test_inds].copy())
                    lbl_th = data_labels[test_inds][: len(out_th)].copy()
                    macro_auroc_t, micro_auprc_t, accuracy_t, f1_t, fb_t, gb_t = evaluate_12ECG_score.compute_metrics(lbl_th, out_th)
                    if fb_t > max_fb_t:
                        print("F-beta score:", fb_t)
                        max_fb_t = fb_t
                        name = "epoch_" + str(epoch) + "_score_" + str(fb_t)
                        saver.save(sess, save_path=model_path + "/" + name)
                    scores.append(fb_t)
                    print("epoch:", epoch, "auroc:", macro_auroc_t, "auprc:", micro_auprc_t, "accuracy:", accuracy_t, "f1:", f1_t, "fb:", fb_t, "gb:", gb_t)
                    summary_metrics = sess.run(metric_summary, feed_dict={f1_t_ph: f1_t, fb_t_ph: fb_t, gb_t_ph: gb_t, macro_auroc_t_ph: macro_auroc_t, micro_auprc_t_ph: micro_auprc_t, f1_v_ph: f1_v, fb_v_ph: fb_v, gb_v_ph: gb_v, macro_auroc_v_ph: macro_auroc_v, micro_auprc_v_ph: micro_auprc_v, accuracy_t_ph: accuracy_t, accuracy_v_ph: accuracy_v})
                    summary_writer.add_summary(summary_metrics, step)

        np.save(log_path + "/scores", scores)
        epoch += 1
