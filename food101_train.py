import os
import tensorflow_datasets as tfds
import tensorflow as tf

# Ensure GPU memory growth is set
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Preprocess the images
def preprocess_image(data):
    image = data['image']
    label = data['label']
    image = tf.image.resize(image, (224, 224))  # Resize images to 224x224
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image, label

# Load and preprocess the dataset
dataset, info = tfds.load('food101', split='train', with_info=True)
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(32)  # Batch size for training
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Load MobileNet model pre-trained on ImageNet
base_model = tf.keras.applications.MobileNet(input_shape=(224, 224, 3),
                                             include_top=False,
                                             weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Add classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(101, activation='softmax')  # 101 classes in Food101 dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=25)  # Adjust epochs as needed

# Ensure the save directory exists
save_dir = 'C:/Tensorflow/models'
os.makedirs(save_dir, exist_ok=True)

# Save the model in the recommended Keras format
model.save(os.path.join(save_dir, 'food101_mobilenet.keras'))

# Save the model in the HDF5 format
model.save(os.path.join(save_dir, 'food101_mobilenet.h5'))

# Save the model as a TensorFlow SavedModel
model.save(os.path.join(save_dir, 'food101_mobilenet'))

# Convert the model to TensorFlow Lite format
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open(os.path.join(save_dir, 'food101_mobilenet.tflite'), 'wb') as f:
        f.write(tflite_model)
    print("Model converted and saved successfully.")
except Exception as e:
    print(f"Error during model conversion: {e}")

# Clean up
del model
tf.keras.backend.clear_session()
