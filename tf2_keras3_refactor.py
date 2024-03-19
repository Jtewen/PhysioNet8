from collections import defaultdict
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
    
    def collect_dataset_info(input_directory):
        names = []
        for f in tqdm(os.listdir(input_directory), "Collecting dataset info"):
            if not f.lower().startswith(".") and f.lower().endswith("hea") and os.path.isfile(os.path.join(input_directory, f)):
                with open(os.path.join(input_directory, f), 'r') as file:
                    first_line = file.readline()
                    name = first_line.split(' ')[0]  # Assuming name is the first word
                    names.append(name)
        data_domains = deep_utils.find_domains(names)
        ref_domains = list(set(data_domains))  # Make sure find_domains works as intended

        return data_domains, ref_domains


# Call this function before initializing your datasets
    data_domains, ref_domains = collect_dataset_info(input_directory)
        
    def load_and_process_data(file_path, class_names, num_classes, Length):
        # Function to read and process each file individually.
        # Returns processed signal, age, sex, label for each record.
        file_path_str = file_path.decode("utf-8")  # Convert bytes to string

        recording, header = deep_utils.load_challenge_data(file_path_str)
        if recording.shape[0] != 12:
            return None  # Or handle error appropriately
        recording = recording.T.astype("float32")
        name = header[0].strip().split(" ")[0]
        try:
            samp_frq = int(header[0].strip().split(" ")[2])
        except Exception as e:
            samp_frq = 500  # Default sampling frequency
        if samp_frq != 500:
            recording = deep_utils.resample(recording.copy(), samp_frq)
        age, sex = 50, 0  # Default values
        label = np.zeros(num_classes, dtype="float32")
        # Parse header for age, sex, and labels
        for l in header:
            if l.startswith("# Age:"):
                age_ = l.strip().split(" ")[2]
                age = 50 if age_ in {"Nan", "NaN"} else int(age_)
            elif l.startswith("# Sex:"):
                sex_ = l.strip().split(" ")[2]
                sex = 0 if sex_ in {"Male", "male", "M"} else 1 if sex_ in {"Female", "female", "F"} else 0
            elif l.startswith("# Dx:"):
                arrs = l.strip().split(" ")
                for arr in arrs[2].split(","):
                    try:
                        class_index = class_names.index(arr.rstrip())
                        label[class_index] = 1.0
                    except:
                        pass
        processed_signal = deep_utils.prepare_data([recording], Length, mod=0)[0]
        domain = deep_utils.find_domains([name])[0]
        return processed_signal.T.astype("float32"), np.float32(age), np.float32(sex), label.T.astype("float32"), domain


    def tf_dataset_from_files(file_paths, class_names, num_classes, Length, batch_size):
        # Create a TF dataset from file paths and map the processing function.
        # Note: Adjust the processing function to return tensors suitable for your model input and labels.
        def load_map_fn(path):
            processed_signal, age, sex, label, domain = tf.numpy_function(
                func=load_and_process_data, 
                inp=[path, class_names, num_classes, Length], 
                Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.uint8]
            )
            # Set shapes
            processed_signal = tf.transpose(processed_signal)
            processed_signal.set_shape((Length, 12))
            age.set_shape(())
            sex.set_shape(())
            label.set_shape((num_classes,))
            domain.set_shape(())  # Single integer representing domain

            # Convert 'domain' to one-hot encoding inside TensorFlow graph
            domain_one_hot = tf.one_hot(domain, depth=len(ref_domains))
            # Structure the data to match the input and output names of your model
            inputs = {'X': processed_signal, 'AGE': age, 'SEX': sex}
            outputs = {'class_output': label, 'domain_output': domain_one_hot}
            return inputs, outputs  # Return structured data
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(load_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Apply batching, prefetching for performance
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    with open("weights.csv") as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [row for row in reader]

    class_names = data_read[0][1:]
    class_weights = np.array(data_read[1:]).astype("float32")[:, 1:]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.save(output_directory + "/class_names.npy", class_names)

    num_classes = len(class_names)
    # Collect header files
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
    # Assuming test_names.npy and val_names.npy contain paths to the respective files
    test_names = np.load("test_val_names/test_names.npy")
    val_names = np.load("test_val_names/val_names.npy")

    # Creating datasets
    num_classes = len(class_names)

    val_size = int(0.2 * len(header_files))  # Assuming 20% for validation
    val_files = header_files[:val_size]
    train_files = header_files[val_size:]

    val_dataset = tf_dataset_from_files(val_files, class_names, num_classes, Length, batch_size)
    train_dataset = tf_dataset_from_files(train_files, class_names, num_classes, Length, batch_size)
    
    

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

    model.compile(optimizer=Adam(1.2e-3),
                  loss={'class_output': 'categorical_crossentropy', 'domain_output': 'categorical_crossentropy'},
                  metrics={'class_output': 'f1_score'})
    
    print("Shape of train_dataset:", train_dataset)

    history = model.fit(train_dataset,
                        epochs=20,
                        batch_size=batch_size,
                        validation_data=val_dataset)


    
    model.save(output_directory + "/model.h5")