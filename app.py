from flask import Flask, render_template, request, redirect, url_for, send_file, session
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
app = Flask(__name__)
app.secret_key = os.urandom(24)  # required for session management
# --- Load Fine-Tuned Model ---
MODEL_PATH = "models/plant_disease_model_finetuned_real.keras"
model = tf.keras.models.load_model(MODEL_PATH)
# --- Class labels ---
class_labels = [
    "Background_without_leaves",
    "Corn_common_rust",
    "Grape_black_rot",
    "Tomato_bacterial_spot",
    "Tomato_yellow_leaf_curl_virus",
    "soyabean_healthy"
]
# --- Disease info with pesticides ---
disease_info = {
    "Background_without_leaves": {
        "cause": "This is background/no leaf detected.",
        "cure": "Please upload a valid plant leaf image for accurate disease detection.",
        "pesticides": "No pesticides needed. Upload a valid leaf for analysis."
    },
    "Corn_common_rust": {
        "cause": "Caused by the fungus Puccinia sorghi; appears as reddish-brown pustules on corn leaves.",
        "cure": "Plant resistant varieties, rotate crops, and apply fungicides if severe.",
        "pesticides": "Apply fungicides such as Mancozeb or Propiconazole; follow recommended dosage."
    },
    "Grape_black_rot": {
        "cause": "Caused by the fungus Guignardia bidwellii; spreads rapidly in warm, humid conditions.",
        "cure": "Remove infected leaves/fruits, use fungicides, and ensure proper ventilation.",
        "pesticides": "Use fungicides like Captan or Myclobutanil to prevent spread; apply during wet season."
    },
    "Tomato_bacterial_spot": {
        "cause": "Caused by Xanthomonas bacteria; spreads through water splashes.",
        "cure": "Use copper-based bactericides, avoid overhead irrigation, plant resistant varieties.",
        "pesticides": "Copper-based sprays (Copper oxychloride or Copper hydroxide) can control bacterial spots."
    },
    "Tomato_yellow_leaf_curl_virus": {
        "cause": "Caused by Tomato Yellow Leaf Curl Virus (TYLCV), transmitted by whiteflies.",
        "cure": "Control whiteflies, remove infected plants, use resistant cultivars.",
        "pesticides": "Use insecticides targeting whiteflies, e.g., Imidacloprid or Neem oil sprays."
    },
    "soyabean_healthy": {
        "cause": "No disease detected, the soybean leaf appears healthy.",
        "cure": "Maintain good crop practices, fertilization, and pest monitoring.",
        "pesticides": "No pesticide required. Monitor regularly for pests."
    }
}
# --- Extra Disease Detail Data with pesticides ---
disease_data = {
    "background": {
        "name": "Background without Leaves",
        "image": "images/background.jpg",
        "symptoms": "No leaf detected in the uploaded image.",
        "cause": "This is background/no leaf detected.",
        "cure": "Please upload a valid plant leaf image for accurate disease detection.",
        "pesticides": "No pesticides needed."
    },
    "corn-rust": {
        "name": "Corn Common Rust",
        "image": "images/corn_common_rust.jpg",
        "symptoms": "Reddish-brown pustules on corn leaves, reduces photosynthesis.",
        "cause": "Caused by the fungus Puccinia sorghi; appears as reddish-brown pustules on corn leaves.",
        "cure": "Plant resistant varieties, rotate crops, and apply fungicides if severe.",
        "pesticides": "Apply fungicides such as Mancozeb or Propiconazole; follow recommended dosage."
    },
    "grape-rot": {
        "name": "Grape Black Rot",
        "image": "images/grape_black_rot.jpg",
        "symptoms": "Brown circular lesions on grapes and leaves; spreads rapidly in humid conditions.",
        "cause": "Caused by the fungus Guignardia bidwellii; spreads rapidly in warm, humid conditions.",
        "cure": "Remove infected leaves/fruits, use fungicides, and ensure proper ventilation.",
        "pesticides": "Use fungicides like Captan or Myclobutanil to prevent spread; apply during wet season."
    },
    "tomato-spot": {
        "name": "Tomato Bacterial Spot",
        "image": "images/tomato_bacterial_spot.jpg",
        "symptoms": "Small, dark, water-soaked spots on tomato leaves and fruits.",
        "cause": "Caused by Xanthomonas bacteria; spreads through water splashes.",
        "cure": "Use copper-based bactericides, avoid overhead irrigation, plant resistant varieties.",
        "pesticides": "Copper-based sprays (Copper oxychloride or Copper hydroxide) can control bacterial spots."
    },
    "tomato-yellow-leaf": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "image": "images/tomato_yellow_leaf_curl.jpg",
        "symptoms": "Upward curling of leaves, yellowing, and stunted plant growth.",
        "cause": "Caused by Tomato Yellow Leaf Curl Virus (TYLCV), transmitted by whiteflies.",
        "cure": "Control whiteflies, remove infected plants, use resistant cultivars.",
        "pesticides": "Use insecticides targeting whiteflies, e.g., Imidacloprid or Neem oil sprays."
    },
    "soybean-healthy": {
        "name": "Soybean Healthy",
        "image": "images/soyabean_healthy.jpg",
        "symptoms": "Leaf appears green, healthy, and free from spots or lesions.",
        "cause": "No disease detected, the soybean leaf appears healthy.",
        "cure": "Maintain good crop practices, fertilization, and pest monitoring.",
        "pesticides": "No pesticide required. Monitor regularly for pests."
    }
}

# --- Preprocess uploaded image ---
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((160, 160))
    img_array = np.array(img)/255.0
    return np.expand_dims(img_array, axis=0)


# --- Login Route (any username, password validation) ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # --- Password Validation ---
        if len(password) < 6:
            error_msg = "Password must be at least 6 characters long."
            return render_template("login.html", error=error_msg)

        if not any(char in password for char in ['@', '#']):
            error_msg = "Password must include at least one special character (@ or #)."
            return render_template("login.html", error=error_msg)

        # ✅ If password is valid, store username in session
        session["username"] = username
        return redirect(url_for("index"))

    return render_template("login.html")



# --- Logout Route ---
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))
# --- Home Page / Index ---
@app.route("/", methods=["GET", "POST"])
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        # Save uploaded file
        filepath = os.path.join("static", file.filename)
        file.save(filepath)
        # Predict
        img_array = preprocess_image(filepath)
        predictions = tf.nn.softmax(model.predict(img_array)).numpy()
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)
        # Get cause & cure
        cause = disease_info[predicted_class]["cause"]
        cure = disease_info[predicted_class]["cure"]
        pesticides = disease_info[predicted_class]["pesticides"] 
        
        return render_template("result.html", 
                               image=file.filename, 
                               disease=predicted_class, 
                               confidence=confidence,
                               cause=cause,
                               cure=cure,
                               pesticides=pesticides,
                               username=session["username"])
    return render_template("index.html", username=session["username"])
# --- Download PDF Route ---
@app.route("/download_pdf/<disease>/<confidence>")
def download_pdf(disease, confidence):
    if "username" not in session:
        return redirect(url_for("login"))
    cause = disease_info[disease]["cause"]
    cure = disease_info[disease]["cure"]
    pesticides = disease_info[disease]["pesticides"]
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, 750, "🌱 Plant Disease Recognition Report")
    c.setFont("Helvetica", 14)
    c.drawString(100, 700, f"Predicted Disease: {disease}")
    c.drawString(100, 670, f"Confidence: {confidence}%")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 630, "Cause:")
    c.setFont("Helvetica", 12)
    c.drawString(120, 610, cause[:90])
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 570, "Cure:")
    c.setFont("Helvetica", 12)
    c.drawString(120, 550, cure[:90])
    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name="prediction_report.pdf",
                     mimetype='application/pdf')
# --- About Page ---
@app.route('/about')
def about():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("about.html", username=session["username"])
# --- Diseases Page ---
@app.route('/diseases')
def diseases():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("diseases.html", username=session["username"], disease_list=disease_data)

# --- Contact Page ---
@app.route('/contact', methods=["GET", "POST"])
def contact():
    if "username" not in session:
        return redirect(url_for("login"))
    success_message = None
    if request.method == "POST":
        # Get submitted data
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        
        # Optionally, handle the message (save/send)
        print(f"Contact Form Submitted: {name}, {email}, {message}")
        # Set success message
        success_message = "Message sent successfully!"
    # Company info
    company_info = {
        "name": "Plant Health Checker Pvt. Ltd.",
        "email": "support@planthealth.com",
        "contact": "+91 9876543210",
        "address": "Plant Health Tower, 123 Greenway Road, Horticulture Park, Nagpur, Maharashtra, India - 440001"
    }
    return render_template("contact.html", 
                           username=session["username"], 
                           success=success_message,
                           company=company_info)
# --- Disease Detail Page ---
@app.route('/disease/<name>')
def disease_detail(name):
    if "username" not in session:
        return redirect(url_for("login"))
    disease = disease_data.get(name)
    if not disease:
        return "Disease not found", 404
    return render_template("disease_detail.html", disease=disease, username=session["username"])

# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)
