import argparse
from glob import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
import deep_utils

LENGTH = 3000

# disable eager execution
tf.compat.v1.disable_eager_execution()

def load_model(input_directory):

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

    tf.compat.v1.reset_default_graph()

    sess = tf.compat.v1.Session()
    new_saver = tf.compat.v1.train.import_meta_graph(selected)
    new_saver.restore(sess, tf.train.latest_checkpoint(input_directory + "/"))

    
    graph = tf.compat.v1.get_default_graph()

    return sess, graph


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

def run_predictions(sess, graph, data_signals, data_ages, data_sexes, batch_size):
    """Generate model predictions on the provided dataset."""
    predictions = []
    n_batches = len(data_signals) // batch_size + (len(data_signals) % batch_size > 0)
    
    for i in range(n_batches):
        batch_signals = data_signals[i * batch_size:(i + 1) * batch_size]
        batch_ages = data_ages[i * batch_size:(i + 1) * batch_size]
        batch_ages = np.reshape(batch_ages, (-1, 1))
        batch_sexes = data_sexes[i * batch_size:(i + 1) * batch_size]
        batch_sexes = np.reshape(batch_sexes, (-1, 1))
        
        # Prepare data using the function from deep_utils
        prepared_signals = prepare_data(batch_signals, LENGTH)
        
        # Get input and output tensors
        X = graph.get_tensor_by_name("X:0")
        AGE = graph.get_tensor_by_name("AGE:0")
        SEX = graph.get_tensor_by_name("SEX:0")
        output_tensor = graph.get_tensor_by_name("out_a:0")
        
        # Run model prediction
        feed_dict = {X: prepared_signals, AGE: batch_ages, SEX: batch_sexes}
        batch_predictions = sess.run(output_tensor, feed_dict=feed_dict)
        predictions.append(batch_predictions)
        
    return np.concatenate(predictions, axis=0)


def find_optimal_thresholds(labels, predictions):
    """Find the optimal threshold for each class based on precision-recall curve."""
    thresholds = []
    for i in range(predictions.shape[1]):  # Assuming predictions and labels are 2D arrays [samples, classes]
        precision, recall, threshold = precision_recall_curve(labels[:, i], predictions[:, i])
        f_scores = 2 * precision * recall / (precision + recall + 1e-6)
        optimal_idx = np.argmax(f_scores)
        thresholds.append(threshold[optimal_idx])
    return np.array(thresholds)

def main(args):
    checkpoint_dir = args.checkpoint_dir    
    # Load the model
    sess, graph = load_model(checkpoint_dir)
    
    # Load data
    data_signals, data_names, data_ages, data_sexes, data_labels = deep_utils.load_pickles("training/processed")
    data_names = np.array(data_names)
    data_ages = np.array(data_ages, dtype="float32")
    data_sexes = np.array(data_sexes, dtype="float32")
    data_labels = np.array(data_labels, dtype="float32")

    data_domains = deep_utils.find_domains(data_names)

    data_domains = np.array(data_domains)
    # Get predictions
    predictions = run_predictions(sess, graph, data_signals, data_ages, data_sexes, batch_size=128)
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(data_labels, predictions)
    
    print("Optimal thresholds:", optimal_thresholds)
    
    # Save the optimal thresholds
    np.save(f"{checkpoint_dir}/optimal_thresholds.npy", optimal_thresholds)
    
    # Optionally, you can now evaluate your model using these thresholds
    # This could involve converting probabilities to labels and comparing with true labels
    # evaluation_score = evaluate_model(labels, predictions, optimal_thresholds)
    # print("Evaluation Score:", evaluation_score)

    sess.close()

if __name__ == '__main__':
    # Model arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', type=str, help='Directory containing the model checkpoint')
    args = parser.parse_args()
    main(args)
