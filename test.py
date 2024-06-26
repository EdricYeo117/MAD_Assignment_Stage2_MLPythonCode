import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the model from the .h5 file
model = tf.keras.models.load_model('C:\\Tensorflow\\models\\food101_mobilenet.h5')

# Alternatively, you can inspect the model summary to see its architecture
model.summary()

# Class names for Food101 dataset
class_names = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", 
               "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", 
               "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", 
               "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla", 
               "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", 
               "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", 
               "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", 
               "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", 
               "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", 
               "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", 
               "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", 
               "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", 
               "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", 
               "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", 
               "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", 
               "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", 
               "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", 
               "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", 
               "tiramisu", "tuna_tartare", "waffles"]

# Function to preprocess a single image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Function to make predictions on a single image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_class]
    return predicted_label

# Directory containing the images to test
test_images_dir = 'C:\\Tensorflow\\test_images'

# Iterate through the images in the test directory and make predictions
for img_file in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_file)
    predicted_label = predict_image(img_path)
    print(f"Image: {img_file}, Predicted Label: {predicted_label}")
