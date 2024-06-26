import tensorflow as tf
import numpy as np

# Load the Keras model
keras_model_path = 'models/food101_mobilenet.keras'
keras_model = tf.keras.models.load_model(keras_model_path)

# Define TensorFlow function
tf_callable = tf.function(
    keras_model.call,
    autograph=False,
    input_signature=[tf.TensorSpec((1, 224, 224, 3), tf.float32)],
)
tf_concrete_function = tf_callable.get_concrete_function()

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tf_concrete_function], tf_callable
)

# Configure converter options
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Define representative dataset for quantization
def representative_dataset_gen():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield data

converter.representative_dataset = representative_dataset_gen

# Convert the model to TensorFlow Lite
tflite_quant_model = converter.convert()

# Save the converted model to a .tflite file
tflite_model_path = 'food101_mobilenet_quant.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f"Model converted to TensorFlow Lite format and saved to: {tflite_model_path}")
