import os
import pickle
import numpy as np
import csv
import deep_utils
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input
from time import time
from tqdm import tqdm
# Import your deep_utils and evaluate_12ECG_score as needed

def build_model(input_shape, num_classes):
    # Primary input (e.g., ECG signals)
    ecg_input = Input(shape=input_shape, name="ecg_input")
    
    # Convolutional layers
    conv1 = layers.Conv1D(48, 9, strides=4, activation='relu')(ecg_input)
    conv2 = layers.Conv1D(64, 7, strides=3, activation='relu')(conv1)
    conv3 = layers.Conv1D(80, 5, strides=2, activation='relu')(conv2)
    conv4 = layers.Conv1D(96, 3, strides=2, activation='relu')(conv3)
    pooled = layers.MaxPooling1D(2, strides=2)(conv4)
    
    # LSTM layer
    lstm_out = layers.Bidirectional(layers.LSTM(128, recurrent_dropout=0.5, return_sequences=False))(pooled)
    
    # Additional inputs
    age_input = Input(shape=(1,), name="age_input")
    sex_input = Input(shape=(1,), name="sex_input")
    
    # Concatenate additional inputs with the LSTM output
    concatenated = layers.concatenate([lstm_out, age_input, sex_input])
    
    # Dense layer for classification
    dense = layers.Dense(100, activation='relu')(concatenated)
    output_layer = layers.Dense(num_classes, activation='sigmoid', name="output")(dense)
    
    model = models.Model(inputs=[ecg_input, age_input, sex_input], outputs=output_layer)
    
    return model

def load_and_preprocess_data(input_directory, class_names, target_length=5600, N_data=10000):
    header_files = (os.path.join(input_directory, f) for f in os.listdir(input_directory)
                    if f.lower().endswith('hea') and not f.lower().startswith('.') and os.path.isfile(os.path.join(input_directory, f)))
    data_signals, data_ages, data_sexes, data_labels = [], [], [], []
    for header_file in tqdm(header_files, desc="Processing files"):
        recording, header_lines = next(deep_utils.load_challenge_data([header_file]))
        fs = int(header_lines[0].split()[2])
        recording = deep_utils.resample(recording, src_frq=fs, trg_frq=500) if fs != 500 else recording
        if recording.shape[1] > target_length:
            recording = recording[:, :target_length]
        else:
            recording = np.pad(recording, ((0, 0), (0, target_length - recording.shape[1])), 'constant', constant_values=0)
        
        age, sex, dx_codes = extract_patient_info(header_lines, class_names)
        data_signals.append(recording)
        data_ages.append(age)
        data_sexes.append(sex)
        data_labels.append(dx_codes)
    return np.array(data_signals), np.array(data_ages), np.array(data_sexes), np.array(data_labels)

def extract_patient_info(header_lines, class_names):
    age = 50  # Default age
    sex = 0  # Male
    dx_codes = np.zeros(len(class_names), dtype="float16")

    for line in header_lines:
        if line.startswith("# Age:"):
            try:
                age = int(line.split(":")[1].strip())
            except ValueError:
                age = 50  # Default age if parsing fails
        elif line.startswith("# Sex:"):
            sex_str = line.split(":")[1].strip()
            sex = 0 if sex_str.lower() == 'male' else 1  # Male or Female
        elif line.startswith("# Dx:"):
            dx_str = line.split(":")[1].strip()
            for dx in dx_str.split(","):
                if dx in class_names:
                    dx_index = class_names.index(dx)
                    dx_codes[dx_index] = 1

    return age, sex, dx_codes


def train_12ECG_classifier(input_directory, output_directory):
    # Assume class_names and class_weights loading remains similar

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

    with open("weights.csv") as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [row for row in reader]
    class_names = data_read[0][1:]
    # Ensure class_weights are properly formatted as a dictionary
    class_weights = {i: weight for i, weight in enumerate(np.array(data_read[1:]).astype("float32")[:, 1:].flatten())}

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_signals, data_ages, data_sexes, data_labels = load_and_preprocess_data(input_directory, class_names)

    # Model configuration
    num_classes = len(class_names)
    Length = 5600  # Ensure all ECG signals have been resized to this length
    
    # Build model
    print("Building model...")
    model = build_model((Length, 12), num_classes)
    
    # Compile model without loss_weights
    print("Compiling model...")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=8e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Prepare dataset for training and validation
    split_idx = int(len(data_signals) * 0.8)  # Example of an 80-20 train-validation split
    X_train, X_val = data_signals[:split_idx], data_signals[split_idx:]
    Y_train, Y_val = data_labels[:split_idx], data_labels[split_idx:]
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=tf.float32)
    # Fit model with class weights applied
    print("Fitting model...")
    model.fit(X_train_tensor, Y_train_tensor,
              validation_data=(X_val, Y_val),
              epochs=100,  # epoch_size
              batch_size=128,  # batch_size
              class_weight=class_weights)  # Apply class weights here

    # Save model
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    model_save_path = os.path.join(output_directory, 'model.h5')
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")