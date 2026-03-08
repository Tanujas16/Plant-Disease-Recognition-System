from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load model
model = tf.keras.models.load_model(
    r"C:\Users\prera\Desktop\Plant_Disease_Recognition\models\plant_disease_model_finetuned.keras"
)

class_labels = [
    'Background_without_leaves', 'Corn_common_rust', 'Grape_black_rot', 
    'Tomato_bacterial_spot', 'Tomato_yello_leaf_curl_virus', 'soyabean_healthy'
]

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((160,160))
    img_array = np.array(img)/255.0
    return np.expand_dims(img_array, axis=0)

# Path to test images folder
test_folder = r"C:\Users\prera\Desktop\Plant_Disease_Recognition\test_folder"

# Loop through all images in the folder
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    img_array = preprocess(img_path)
    pred = tf.nn.softmax(model.predict(img_array)).numpy()
    predicted_class = class_labels[np.argmax(pred)]
    confidence = round(100 * np.max(pred), 2)
    print(f"Image: {img_name} → Predicted class: {predicted_class}, Confidence: {confidence}%")
