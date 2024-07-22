import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# Ensure GPU memory growth is set
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
                
# Load and preprocess the dataset
def preprocess_image(data):
    image = data['image']
    label = data['label']
    image = tf.image.resize(image, (224, 224))  # Resize images to 224x224
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image, label

dataset, info = tfds.load('food101', split='train', with_info=True)
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(32)  # Batch size for training
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Split dataset into training and validation sets
train_size = int(0.8 * info.splits['train'].num_examples)
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(101, activation='softmax')  # 101 classes in Food101 dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=1, validation_data=validation_dataset)

# Save the model
model.save('my_unique_food101_model.keras')

# Save the model in the HDF5 format
model.save('my_unique_food101_model.h5')

# Save the model as a TensorFlow SavedModel
model.save('my_unique_food101_model')

# Convert the model to TensorFlow Lite format
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open('my_unique_food101_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted and saved successfully.")
except Exception as e:
    print(f"Error during model conversion: {e}")

# Clean up
del model
tf.keras.backend.clear_session()
