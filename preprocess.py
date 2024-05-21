import csv
import os
import numpy as np
from tqdm import tqdm
import deep_utils
import pickle

def preprocess_and_save_data(input_directory, output_directory):
    with open("weights.csv") as fp:
        reader = csv.reader(fp, delimiter=",")
        data_read = [row for row in reader]

    class_names = data_read[0][1:]

    num_classes = len(class_names)
    print("Loading data from files...")
    header_files = []
    for f in tqdm(os.listdir(input_directory), "Preparing data..."):
        g = os.path.join(input_directory, f)
        if (
            not f.lower().startswith(".")
            and f.lower().endswith("hea")
            and os.path.isfile(g)
        ):
            header_files.append(g)
    
    num_files = len(header_files)
    print("num files", num_files)

    data_names, data_ages, data_sexes, data_labels, data_signals = [], [], [], [], []

    header_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) 
                    if not f.lower().startswith(".") and f.lower().endswith("hea") and os.path.isfile(os.path.join(input_directory, f))]

    for i in tqdm(range(num_files), "Processing data..."):
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

        if samp_frq != 257:
            recording = deep_utils.resample(recording.copy(), samp_frq, 257)

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

        data_names.append(name)
        data_ages.append(age)
        data_sexes.append(sex)
        data_labels.append(label)
        data_signals.append(recording)  # Transpose back to original shape if necessary
        
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # pickle and save data
    with open(os.path.join(output_directory, "data_signals.pkl"), "wb") as f:
        pickle.dump(data_signals, f)
    with open(os.path.join(output_directory, "data_names.pkl"), "wb") as f:
        pickle.dump(data_names, f)
    with open(os.path.join(output_directory, "data_ages.pkl"), "wb") as f:
        pickle.dump(data_ages, f)
    with open(os.path.join(output_directory, "data_sexes.pkl"), "wb") as f:
        pickle.dump(data_sexes, f)
    with open(os.path.join(output_directory, "data_labels.pkl"), "wb") as f:
        pickle.dump(data_labels, f)

    print("Preprocessing complete. Data saved to:", output_directory)

if __name__ == "__main__":
    input_directory = "training/"
    output_directory = "training/processed/"

    preprocess_and_save_data(input_directory, output_directory)