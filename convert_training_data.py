import json
import numpy as np
from transformers import BertTokenizer

# Load the train dataset
with open('train.json') as f:
    train_data = json.load(f)

# Function to create sentences with annotated food entities
def annotate_food_entities(recipes):
    annotated_sentences = []
    for recipe in recipes:
        for ingredient in recipe['ingredients']:
            sentence = f"I want to cook {ingredient}."
            annotated_sentences.append({"sentence": sentence, "ingredient": ingredient})
    return annotated_sentences

# Annotate train data
train_annotated_data = annotate_food_entities(train_data)

# Tokenize sentences using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer([item['sentence'] for item in train_annotated_data], 
                            truncation=True, padding='max_length', max_length=64, return_tensors='tf')

# Convert labels to integer-encoded arrays
unique_ingredients = list(set(item['ingredient'] for item in train_annotated_data))
ingredient_to_index = {ingredient: index for index, ingredient in enumerate(unique_ingredients)}
np.save('ingredient_to_index.npy', ingredient_to_index)
train_labels = np.array([ingredient_to_index[item['ingredient']] for item in train_annotated_data])

# Save preprocessed training data
np.savez('train_data.npz',
         input_ids=train_encodings.input_ids.numpy(),
         attention_masks=train_encodings.attention_mask.numpy(),
         token_type_ids=train_encodings.token_type_ids.numpy(),
         labels=train_labels,
         ingredient_to_index=ingredient_to_index)

# Print a few examples to verify
print(train_annotated_data[:5])

