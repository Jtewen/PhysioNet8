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



def train_12ECG_classifier(input_directory, output_directory):

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
    for i in tqdm(range(num_files), "Loading data"):
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
                if sex_ == "Male" or sex_ == "male" or sex_ == "M":
                    sex = 0
                elif sex_ == "Female" or sex_ == "female" or sex_ == "F":
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
        processed_signal = deep_utils.prepare_data([recording], Length, mod=0)[0]
        data_signals.append(processed_signal)
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
        
    def create_dataset(signals, ages, sexes, labels, batch_size):
        # Convert data to tensors and preprocess ages and sexes
        signals = tf.constant(signals, dtype=tf.float32)
        ages = tf.constant(ages, dtype=tf.float32)[:, tf.newaxis] / 100.0
        sexes = tf.constant(sexes, dtype=tf.float32)[:, tf.newaxis]
        labels = tf.constant(labels, dtype=tf.float32)

        # Combine into a single dataset
        dataset = tf.data.Dataset.from_tensor_slices(({"X": signals, "AGE": ages, "SEX": sexes}, labels))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Prefetching for performance
        return dataset

    



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
        
        
        # Assuming data_signals, data_ages, data_sexes, data_labels are numpy arrays
    train_dataset = create_dataset(
        signals=np.array(data_signals)[train_inds],
        ages=np.array(data_ages)[train_inds],
        sexes=np.array(data_sexes)[train_inds],
        labels=np.array(data_labels)[train_inds],
        batch_size=batch_size
    )

    val_dataset = create_dataset(
        signals=np.array(data_signals)[val_inds],
        ages=np.array(data_ages)[val_inds],
        sexes=np.array(data_sexes)[val_inds],
        labels=np.array(data_labels)[val_inds],
        batch_size=batch_size
    )

    test_dataset = create_dataset(
        signals=np.array(data_signals)[test_inds],
        ages=np.array(data_ages)[test_inds],
        sexes=np.array(data_sexes)[test_inds],
        labels=np.array(data_labels)[test_inds],
        batch_size=batch_size
    )
    
    train_dataset_cardinality = tf.data.experimental.cardinality(train_dataset).numpy()
    val_dataset_cardinality = tf.data.experimental.cardinality(val_dataset).numpy()
    test_dataset_cardinality = tf.data.experimental.cardinality(test_dataset).numpy()

    print("Train dataset cardinality:", train_dataset_cardinality)
    print("Validation dataset cardinality:", val_dataset_cardinality)
    print("Test dataset cardinality:", test_dataset_cardinality)

    ##########################

    print("Elapsed:", time() - t0)
    print("Building model...")
    
    def create_model(input_shape, num_classes, dropout_rate, num_domains):
        # Inputs
        X_input = Input(shape=input_shape, name="X")
        AGE_input = Input(shape=(1,), name="AGE")
        SEX_input = Input(shape=(1,), name="SEX")
        
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
        
        # Dense Layers for Domain Adaptation
        domain_adaptation = Dense(64, activation="relu")(concatenated)
        domain_adaptation = Dropout(dropout_rate)(domain_adaptation)
        domain_output = Dense(num_domains, activation="softmax", name="domain_output")(domain_adaptation)
        
        model = Model(inputs=[X_input, AGE_input, SEX_input], outputs=[out_a, domain_output])
        
        return model
    
    model = create_model(input_shape=(Length, 12), num_classes=len(class_names), dropout_rate=0.5, num_domains=len(ref_domains))

    model.compile(optimizer='adam',
                  loss={'class_output': 'binary_crossentropy', 'domain_output': 'categorical_crossentropy'},
                  metrics={'class_output': 'accuracy'})
    
    for batch in train_dataset.take(1):
        features, labels = batch
        print("Type of features:", type(features))
        print("Type of labels:", type(labels))
        print("Shape of features:", features['X'].shape)
        print("Shape of labels:", labels.shape)
    

    tf.config.run_functions_eagerly(True)
        # Convert datasets to eager tensors to access cardinality
    train_dataset_eager = train_dataset.as_numpy_iterator()
    val_dataset_eager = val_dataset.as_numpy_iterator()
    # Calculate cardinality of train_dataset
    train_dataset_cardinality = len(list(train_dataset_eager))
    # Calculate cardinality of val_dataset
    val_dataset_cardinality = len(list(val_dataset_eager))
    # Print the cardinalities
    print("Cardinality of train_dataset:", train_dataset_cardinality)
    print("Cardinality of val_dataset:", val_dataset_cardinality)
    # Disable eager execution after calculating cardinalities
    # Now you can proceed with model training
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=100,
                        batch_size=128,
                        steps_per_epoch=train_dataset_cardinality,
                        validation_steps=val_dataset_cardinality)

    
    model.save(output_directory + "/model.h5")