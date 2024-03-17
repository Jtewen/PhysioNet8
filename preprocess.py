import csv
import os
import numpy as np
from tqdm import tqdm
import deep_utils
import pickle  # Import pickle for serialization

def preprocess_and_save_data(input_directory, target_length=5600, batch_size=100):
    with open("weights.csv") as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [row for row in reader]
    class_names = data_read[0][1:]
    header_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory)
                    if f.lower().endswith('hea') and not f.lower().startswith('.') and os.path.isfile(os.path.join(input_directory, f))]
    
    dict_path = "training/preprocessed_data"
    os.makedirs(dict_path, exist_ok=True)

    batch_data = []
    batch_counter = 0

    for header_file in tqdm(header_files, desc="Preprocessing"):
        recording, header_lines = next(deep_utils.load_challenge_data([header_file]))
        fs = int(header_lines[0].split()[2])
        recording = deep_utils.resample(recording, src_frq=fs, trg_frq=500) if fs != 500 else recording

        # Ensure recording is correctly sized (5600 time points, 12 leads)
        if recording.shape[1] > target_length:
            recording = recording[:, :target_length]
        elif recording.shape[1] < target_length:
            padding = target_length - recording.shape[1]
            recording = np.pad(recording, ((0, 0), (0, padding)), 'constant', constant_values=0)

        # Verify recording shape matches expected input for prepare_data
        assert recording.shape == (12, target_length), f"Recording shape mismatch: {recording.shape}"

        # Reshape for prepare_data (expects (samples, time points, features))
        recording = recording.T.reshape(1, target_length, 12)  # Transpose to (time points, leads), then reshape

        age, sex, dx_codes = extract_patient_info(header_lines, class_names)

        # Process recording with prepare_data
        processed_recording = deep_utils.prepare_data(recording, target_length, aug=True)[0]  # Process and extract the first item

        batch_data.append((processed_recording, age, sex, dx_codes))
        if len(batch_data) >= batch_size:
            with open(os.path.join(dict_path, f"batch_{batch_counter}.pkl"), 'wb') as f:
                pickle.dump(batch_data, f)
            batch_data = []
            batch_counter += 1

    if batch_data:  # Save any remaining data
        with open(os.path.join(dict_path, f"batch_{batch_counter}.pkl"), 'wb') as f:
            pickle.dump(batch_data, f)

    print("Preprocessing complete.")

def extract_patient_info(header_lines, class_names):
    age = 50  # Default age
    sex = 0  # Male
    dx_codes = np.zeros(len(class_names), dtype="float32")  # Ensure consistency in dtype

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

if __name__ == "__main__":
    preprocess_and_save_data("training")
