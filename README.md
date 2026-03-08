# Plant Disease Recognition System

**Short description**

This project is a web-based Plant Disease Recognition System built with TensorFlow / Keras and Flask. It loads a trained CNN model to classify leaf images into disease categories and provides a simple web UI for uploading images and viewing results. The repository included a ready-to-use fine-tuned model so you can run inference locally right away.

---

## Features
- Web UI (Flask) for uploading plant leaf images and viewing predictions.
- Pretrained / fine-tuned Keras model included (`models/plant_disease_model_finetuned.keras`, `models/plant_disease_model_finetuned_real.keras`).
- Example test script for batch inference (`test_model.py`).
- Static demo images and HTML templates for results and disease information.

---

## Quick repository overview (files & folders)
```
Plant_Disease_Recognition/
├─ app.py                    # Flask web application (serves UI + inference)
├─ test_model.py             # Simple script to run predictions on a test folder
├─ requirements.txt          # Python dependencies
├─ models/                   # Saved Keras models (.keras files)
│   ├─ plant_disease_model_finetuned.keras
│   └─ plant_disease_model_finetuned_real.keras
├─ templates/                # HTML templates for the web UI
├─ static/                   # CSS, images, and other static assets
└─ test_folder/              # A few sample images included for quick testing
```

> Note: `test_model.py` in this repo contains absolute Windows paths. See **Running tests** below for how to run it on your machine.

---

## Requirements
- Python 3.10 or 3.11 recommended (TensorFlow 2.19 has narrower support for Python versions; if you use a different Python version check compatibility).
- Recommended hardware: CPU-only will work for inference, GPU recommended for training.

**Python packages** (install from `requirements.txt`):
```
Flask==3.0.3
tensorflow==2.19.0
keras==3.10.0
numpy
Pillow
```
Install with:
```bash
# from project root
python -m venv venv
# linux / mac
source venv/bin/activate
# windows (PowerShell)
# .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**GPU notes**
- If you want to run/train on GPU, ensure you have a compatible CUDA and cuDNN version for TensorFlow 2.19.0 and the proper NVIDIA drivers installed.

---

## How to run the web app (step-by-step)
1. Unzip or clone the repository and `cd` into `Plant_Disease_Recognition`.
2. Create and activate a virtual environment and install the requirements (see previous section).
3. Confirm the model file exists at `models/plant_disease_model_finetuned_real.keras` (the Flask app `app.py` uses this relative path by default). If you want to use the other model (`plant_disease_model_finetuned.keras`) update the `MODEL_PATH` constant inside `app.py`.
4. Start the Flask server:
```bash
python app.py
```
By default the app runs on `http://127.0.0.1:5000/`. Open that in your browser.

5. Use the web UI to upload a leaf image and view predicted disease + confidence.

---

## Running the example batch tester (`test_model.py`)
`test_model.py` demonstrates how to load a saved Keras model and run predictions on all images in a folder. The script currently uses absolute Windows paths (e.g. `C:\Users\...`). To run it on your machine:

1. Open `test_model.py` and change the `model = tf.keras.models.load_model(...)` path to the relative path inside the repo, for example:
```py
model = tf.keras.models.load_model("models/plant_disease_model_finetuned.keras")
```
2. Update `test_folder` to point to `test_folder/` in the repo or your own folder of images:
```py
test_folder = os.path.join(os.path.dirname(__file__), "test_folder")
```
3. Run:
```bash
python test_model.py
```
You should see printed lines like:
```
Image: tomato_bacterial_spot.JPG → Predicted class: Tomato_bacterial_spot, Confidence: 98.23%
```

---

## Techniques used to create / train the model (summary)
> The repository includes a fine-tuned model (`*_finetuned*.keras`). The precise training script is not included; below is a concise summary of the common techniques used for this kind of plant-disease classifier and a recommended training recipe if you want to re-train or further fine-tune the model yourself.

**Likely techniques / best practices used**
- **Transfer learning / fine-tuning:** starting from a CNN backbone pre-trained on ImageNet (e.g., MobileNetV2, ResNet50, EfficientNet) and adapting the top layers for the plant disease classes.
- **Data preprocessing:** resizing images to a fixed size (e.g. 224×224 or 256×256), scaling pixel values to [0,1], and one-hot encoding class labels.
- **Data augmentation:** random flips, rotations, zoom, brightness adjustments to increase robustness to variations.
- **Loss & optimizer:** categorical crossentropy with optimizers like Adam; using metrics such as accuracy and F1-score for imbalanced classes.
- **Training best-practices:** callbacks including `ModelCheckpoint`, `EarlyStopping`, and learning-rate scheduling.

**Recommended quick training recipe (Keras)**
```py
# high-level sketch (not included as a runnable script in repo)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False  # start with frozen base
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(base.input, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator or tf.data for augmentation
# Train for N epochs, unfreeze some base layers and fine-tune with low LR
```

---

## Model files & labels
- The project contains two saved models in `models/`:
  - `plant_disease_model_finetuned.keras`
  - `plant_disease_model_finetuned_real.keras` (used by `app.py` by default)

- `test_model.py` defines the class label list (edit if you use a different dataset):
```
class_labels = [
  'Background_without_leaves', 'Corn_common_rust', 'Grape_black_rot',
  'Tomato_bacterial_spot', 'Tomato_yello_leaf_curl_virus', 'soyabean_healthy'
]
```
Update this list if you change model or dataset.

---

## Troubleshooting & tips
- **TensorFlow errors on install**: ensure your Python version is compatible with the TensorFlow wheel (TF 2.19.0 supports Python 3.10/3.11). If you need GPU support, install CUDA and cuDNN versions that match TF 2.19.0.
- **Model not loading**: check the `MODEL_PATH` inside `app.py` and ensure the `.keras` file is present and not corrupted.
- **Slow inference**: resize images to the model’s expected input size; for production consider converting the model to TensorFlow SavedModel or TFLite, or use a GPU.
- **Permission / port in use**: if `127.0.0.1:5000` is in use, change the port in `app.run(debug=True)` to `app.run(debug=True, port=5001)` or similar.

---
Repository Link :
https://github.com/Tanujas16/Plant-Disease-Recognition-System





