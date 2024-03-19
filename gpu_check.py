import tensorflow as tf

# Check if GPUs are available
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPUs are available:")
    for gpu in gpus:
        print("Name:", gpu.name, "Type:", gpu.device_type)
else:
    print("No GPUs found. Make sure you have CUDA and cuDNN installed.")
