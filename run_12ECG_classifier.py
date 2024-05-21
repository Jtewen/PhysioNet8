import numpy as np
import tensorflow as tf
from glob import glob


def resample(data, src_frq, trg_frq=500):
    N_src = data.shape[0]
    N_trg = int(N_src * trg_frq / src_frq)

    resampled = np.zeros((N_trg, data.shape[1]), dtype="float32")
    for i in range(data.shape[1]):
        resampled[:, i] = np.interp(
            np.linspace(0, N_src, N_trg), np.arange(N_src), data[:, i]
        )

    return resampled


def find_st_end(data_len, final_len):

    st = 0
    end = data_len

    great = max(data_len, final_len)
    small = min(data_len, final_len)

    if great != small:
        diff = great - small

        st = np.random.randint(0, diff + 1)
        end = st + small

    return st, end


def prepare_data(x, length, mod=0, scale=1000.0, clip1=-100.0, clip2=100.0):

    data = np.zeros((len(x), length, 12))
    for i in range(len(x)):
        sig = x[i][mod::2].copy()
        L = sig.shape[0]
        st, end = find_st_end(L, length)
        if L > length:
            data[i] = sig[st:end]
        else:
            data[i, st:end] = sig

    data = np.clip(data / scale, clip1, clip2)

    return data


def run_12ECG_classifier(data, header_data, loaded_model):

    Length = 3000

    if data.shape[0] != 12:
        print("Error in number of leads!", data.shape)

    data = data.T.astype("float32")

    name = header_data[0].strip().split(" ")[0]

    try:
        samp_frq = int(header_data[0].strip().split(" ")[2])
    except:
        print("Error while reading sampling frequency!", header_data[0])
        samp_frq = 500

    if samp_frq != 257:
        data = resample(data.copy(), samp_frq, 257)

    age = 50
    sex = 0

    for l in header_data:

        if l.startswith("#Age:"):
            age_ = l.strip().split(" ")[1]

            if age_ == "Nan" or age_ == "NaN":
                age = 50
            else:
                try:
                    age = int(age_)
                except:
                    print("Error in reading age!", age_)
                    age = 50

        if l.startswith("#Sex:"):
            sex_ = l.strip().split(" ")[1]
            if sex_ == "Male" or sex_ == "male":
                sex = 0
            elif sex_ == "Female" or sex_ == "female":
                sex = 1
            else:
                print("Error in reading sex!", sex_)

    sex = np.array([[sex]])
    age = np.array([[age]]) / 100.0

    threshold = loaded_model["threshold"]
    sess = loaded_model["session"]

    data_in = prepare_data(data[np.newaxis], Length, mod=np.random.randint(0, 2))

    outs = sess.run("out_a:0", {"X:0": data_in, "AGE:0": age, "SEX:0": sex})

    current_score = np.mean(outs, 0)
    pred = current_score > threshold
    current_label = pred.astype(int)
    if current_label.sum() < 1:
        current_label[np.argmax(current_score)] = 1

    return current_label, current_score, loaded_model["classes"]


def load_12ECG_model(input_directory):

    files = glob(input_directory + "/*.meta")
    max_score = -1
    for file in files:
        try:
            score = int(file[-7:-5])
        except:
            score = 0

        if score > max_score:
            max_score = score
            selected = file

    model = {}

    model["classes"] = np.load(input_directory + "/class_names.npy")
    model["threshold"] = np.load(input_directory + "/optimal_thresholds.npy")

    tf.compat.v1.reset_default_graph()

    sess = tf.compat.v1.Session()
    new_saver = tf.compat.v1.train.import_meta_graph(selected)
    new_saver.restore(sess, tf.train.latest_checkpoint(input_directory + "/"))

    model["session"] = sess

    return model
