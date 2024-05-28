from collections import Counter, defaultdict
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
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, LSTM, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric
from keras import backend as K
from functools import partial
from tensorflow.keras.callbacks import Callback






def train_12ECG_classifier_tf2(input_directory, output_directory):

    seed = 0
    np.random.seed(seed)

    epoch_size = 100

    learning_rate = 8e-4
    Length = 5600

    lambdaa_d = 1e-1

    lambdaa_DG = 1e-3

    batch_size = 128

    n_step = 80
    
    keep_prob = 0.5  # as a static value
    dropout_rate = 1 - keep_prob

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
    
    resampled_signals = []

    for signal in data_signals:
        resampled_signal = deep_utils.resample(signal, 500, 257)
        resampled_signals.append(resampled_signal)

    data_signals = resampled_signals
    
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
    training_domains = ref_domains.copy()
    training_domains.remove(4)
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
        
    np.save(output_directory + "/class_names.npy", class_names)

    num_classes = len(class_names)
    print("num classes", num_classes)

    # Assuming test_names.npy and val_names.npy contain paths to the respective files
    test_names = np.load("test_val_names/test_names.npy")
    val_names = np.load("test_val_names/val_names.npy")

    # Creating datasets
    num_classes = len(class_names)
    
    def make_dataset(indices, fetch_data, batch_size):
        def generator():
            Xs, AGEs, SEXs, class_outputs, domain_outputs, Y_Ws, D_Ms, Y_Ds = [], [], [], [], [], [], [], []
            inds = []
            for ind in indices:
                inds.append(ind)
                data = fetch_data(ind)
                Xs.append(data[0]['X'])
                AGEs.append(data[0]['AGE'])
                SEXs.append(data[0]['SEX'])
                class_outputs.append(data[1]['class_output'])
                domain_outputs.append(data[1]['domain_output'])
                Y_Ds.append(data[1]['Y_D'])
                if len(Xs) >= batch_size:
                    Xs = deep_utils.prepare_data(Xs, Length)
                    Y_W = deep_utils.calc_weights(class_outputs[:batch_size], class_weights)
                    D_M = domain_masks[inds[:batch_size]].copy()
                    yield {"X": Xs, "AGE": np.array(AGEs), "SEX": np.array(SEXs)}, {"class_output": np.array(class_outputs), "domain_output": np.array(domain_outputs), "Y_W": np.array(Y_W), "D_M": np.array(D_M), "Y_D": np.array(Y_Ds)}
                    Xs, AGEs, SEXs, class_outputs, domain_outputs, Y_Ws, D_Ms, Y_Ds = [], [], [], [], [], [], [], []
    
        return tf.data.Dataset.from_generator(generator, output_signature=(
            {
                "X": tf.TensorSpec(shape=(None, None, 12), dtype=tf.float32),
                "AGE": tf.TensorSpec(shape=(None,1), dtype=tf.float32),
                "SEX": tf.TensorSpec(shape=(None,1), dtype=tf.float32)
            },
            {
                "class_output": tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
                "domain_output": tf.TensorSpec(shape=(None,len(ref_domains)), dtype=tf.float32),
                "Y_W": tf.TensorSpec(shape=(None,num_classes), dtype=tf.float32),
                "D_M": tf.TensorSpec(shape=(None,num_classes), dtype=tf.float32),
                "Y_D": tf.TensorSpec(shape=(None,len(ref_domains)), dtype=tf.float32)
            }
        ))
        
    def fetch_data(ind):
        return (
            {
                "X": data_signals[ind],
                "AGE": data_ages[ind],
                "SEX": data_sexes[ind]
            },
            {
                "class_output": data_labels[ind],
                "domain_output": data_domains[ind],
                "Y_D": data_domains[ind]
            }
        )

    train_dataset = make_dataset(train_inds, fetch_data, batch_size)
    val_dataset = make_dataset(val_inds, fetch_data, len(val_inds))
    test_dataset = make_dataset(test_inds, fetch_data, len(test_inds))

    logdir = "tensorboard/" + output_directory
    summary_writer = tf.summary.create_file_writer(logdir)
    ##########################

    print("Elapsed:", time() - t0)
    print("Building model...")
    
    class GradientReversal(tf.keras.layers.Layer):
        def __init__(self, lambdaa_DG):
            super(GradientReversal, self).__init__()
            self.lambdaa_DG = lambdaa_DG

        def call(self, inputs):
            return inputs

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'lambdaa_DG': self.lambdaa_DG,
            })
            return config

        @tf.custom_gradient
        def grad_reverse(x, lambdaa_DG):
            y = tf.identity(x)
            def custom_grad(dy):
                return -lambdaa_DG * dy, None
            return y, custom_grad
    
    def create_model(input_shape, num_classes, dropout_rate, num_domains):
        # Inputs
        AGE_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="AGE")
        SEX_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="SEX")
        X_input = tf.keras.Input(shape=(Length, 12), dtype=tf.float32, name="X")

        # Outputs and weights
        Y_input = tf.keras.Input(shape=(num_classes,), dtype=tf.float32, name="Y")
        Y_W_input = tf.keras.Input(shape=(num_classes,), dtype=tf.float32, name="Y_W")
        D_M_input = tf.keras.Input(shape=(num_classes,), dtype=tf.float32, name="D_M")
        Y_D_input = tf.keras.Input(shape=(len(ref_domains),), dtype=tf.float32, name="Y_D")
                
        
        # First Convolutional Block
        e1 = Conv1D(filters=48, kernel_size=9, strides=4, padding="valid", activation="relu")(X_input)
        e2 = Conv1D(filters=64, kernel_size=7, strides=3, activation="relu")(e1)
        e3 = Conv1D(filters=80, kernel_size=5, strides=2, activation="relu")(e2)
        e4 = Conv1D(filters=96, kernel_size=3, strides=2, activation="relu")(e3)
        e5_0 = MaxPooling1D(pool_size=2, strides=2)(Conv1D(filters=112, kernel_size=3, strides=2, activation="relu")(e4))
        
        # LSTM Block
        emb1 = Bidirectional(LSTM(128, return_sequences=False))(e5_0)
        
        # Second Convolutional Block (potentially with masking)
        # Assuming you want to replicate the conv structure but with a different kernel size as per your example
        e1_alt = Conv1D(filters=48, kernel_size=19, strides=4, padding="valid", activation="relu")(X_input)
        e2_alt = Conv1D(filters=64, kernel_size=15, strides=3, activation="relu")(e1_alt)
        e3_alt = Conv1D(filters=80, kernel_size=11, strides=2, activation="relu")(e2_alt)
        e4_alt = Conv1D(filters=96, kernel_size=9, strides=2, activation="relu")(e3_alt)
        e5_0_alt = MaxPooling1D(pool_size=2, strides=2)(Conv1D(filters=112, kernel_size=7, strides=2, activation="relu")(e4_alt))
        emb2 = Bidirectional(LSTM(128, return_sequences=False))(e5_0_alt)
        
        # Concatenation
        concatenated = Concatenate(axis=1)([emb1, emb2, AGE_input, SEX_input])
        
        # Dense Layers for Classification
        x = Dense(100, activation="relu")(concatenated)
        x = Dropout(dropout_rate)(x)
        out_a = Dense(num_classes, activation="sigmoid", name="class_output")(x)
        
        # Dense Layers for Domain Adaptation with gradient reversal layer
        grl = GradientReversal(lambdaa_DG)(concatenated)
        domain_adaptation = Dense(64, activation="relu")(grl)
        domain_adaptation = Dropout(dropout_rate)(domain_adaptation)
        domain_output = Dense(num_domains, activation="sigmoid", name="domain_output")(domain_adaptation)
        
        model = Model(inputs=[X_input, AGE_input, SEX_input, ], outputs=[out_a, domain_output])
                
        return model
    
    model = create_model(input_shape=(Length, 12), num_classes=len(class_names), dropout_rate=0.5, num_domains=len(ref_domains))
    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam()
    
    def masked_sigmoid_cross_entropy(y_true, y_pred, mask):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        masked_loss = tf.multiply(loss, mask)
        return tf.reduce_mean(masked_loss)
    
    # Define the loss functions (sigmoid cross-entropy with logits)
    class_loss_fn = masked_sigmoid_cross_entropy
    domain_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    class CustomMetric(tf.keras.metrics.Metric):
        def __init__(self, num_classes, class_weights, class_names, step_size=0.01, name="custom_metric", **kwargs):
            super().__init__(name=name, **kwargs)
            self.class_weights = class_weights
            self.class_names = class_names
            self.thresholds = self.add_weight(name="thresholds", shape=(num_classes,), initializer=tf.keras.initializers.Constant(0.1))
            self.total = self.add_weight(name="total", initializer="zeros")
            self.step_size = tf.fill((num_classes,), step_size)
    
        def update_state(self, y_true, y_pred, sample_weight=None):
            # Apply the thresholds to y_pred
            y_pred_thresholded = tf.cast(y_pred >= self.thresholds, y_true.dtype)
    
            def compute_metric(y_true, y_pred_thresholded):
                return evaluate_12ECG_score.compute_challenge_metric(self.class_weights, y_true, y_pred_thresholded, list(self.class_names), "426783006")
    
            metric_value = tf.py_function(compute_metric, [y_true, y_pred_thresholded], tf.float32)
            self.total.assign_add(metric_value)
    
            # Check the score at the current thresholds, and at 0.01 higher and lower
            scores = []
            for delta in [-self.step_size, 0, self.step_size]:
                y_pred_thresholded = tf.cast(y_pred >= (self.thresholds + delta), y_true.dtype)
    
                score = tf.py_function(compute_metric, [y_true, y_pred_thresholded], tf.float32)
                scores.append(score)
    
            # Move in the direction that increases the score
            if scores[0] > scores[1]:
                self.thresholds.assign_sub(self.step_size)
            elif scores[2] > scores[1]:
                self.thresholds.assign_add(self.step_size)
    
        def result(self):
            return self.total.value()
    
        def reset_states(self):
            self.total.assign(0.0)
            
    val_metric = CustomMetric(num_classes, class_weights, class_names)
    test_metric = CustomMetric(num_classes, class_weights, class_names)
        
    @tf.function
    def train_step(X, AGE, SEX, Y, Y_W, D_M, Y_D, i):
        
        with tf.GradientTape() as tape:
            # Forward pass
            class_output, domain_output = model([X, AGE, SEX])
    
            # Calculate the mask
            MASK = Y_W * D_M
            # Calculate the losses
            class_loss = class_loss_fn(Y, class_output, MASK)
            domain_loss = domain_loss_fn(Y_D, domain_output)
    
            # Calculate the total loss
            total_loss = class_loss + 0.1 * domain_loss
    
        # Calculate the gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)
    
        # Update the weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Log the metrics
        with summary_writer.as_default():
            tf.summary.scalar("Loss", class_loss, step=i)
            tf.summary.scalar("Domain_Loss", domain_loss, step=i)
        

            
    def evaluate(X, AGE, SEX, Y, Y_W, D_M, Y_D, metric):
        class_output, domain_output = model([X, AGE, SEX])
        metric.update_state(Y, class_output)
        
    
    # Training loop
    for epoch in range(epoch_size):
        print(f"Epoch {epoch + 1}/{epoch_size}")
        for i, batch in enumerate(train_dataset):
            inputs, labels = batch
            X, AGE, SEX = inputs['X'], inputs['AGE'], inputs['SEX']
            Y, Y_W, D_M, Y_D = labels['class_output'], labels['Y_W'], labels['D_M'], labels['Y_D']
            
            train_step(X, AGE, SEX, Y, Y_W, D_M, Y_D, epoch * len(train_inds) + i)
            
            # Evaluate the model every n_step batches
            if i % n_step == 0:
                #get current batch of validation data
                val_batch = next(iter(val_dataset))
                inputs_val, labels_val = val_batch
                X_val, AGE_val, SEX_val = inputs_val['X'], inputs_val['AGE'], inputs_val['SEX']
                Y_val, Y_W_val, D_M_val, Y_D_val = labels_val['class_output'], labels_val['Y_W'], labels_val['D_M'], labels_val['Y_D']
                
                evaluate(X_val, AGE_val, SEX_val, Y_val, Y_W_val, D_M_val, Y_D_val, val_metric)
                
                #get current batch of test data
                test_batch = next(iter(test_dataset))
                inputs_test, labels_test = test_batch
                X_test, AGE_test, SEX_test = inputs_test['X'], inputs_test['AGE'], inputs_test['SEX']
                Y_test, Y_W_test, D_M_test, Y_D_test = labels_test['class_output'], labels_test['Y_W'], labels_test['D_M'], labels_test['Y_D']
                evaluate(X_test, AGE_test, SEX_test, Y_test, Y_W_test, D_M_test, Y_D_test, test_metric)
                # Log the metrics
                with summary_writer.as_default():
                    tf.summary.scalar("val_score", val_metric.result(), step=i)
                    tf.summary.scalar("test_score", test_metric.result(), step=i)
            
        # Log the epoch
        print(f"Validation score: {val_metric.result().numpy()}")
        print(f"Test score: {test_metric.result().numpy()}")
        

    model.save(output_directory + "/model.keras")