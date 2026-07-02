from flask import Flask, request, render_template, redirect, url_for, session
import os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for session

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory "database" for demo (replace with real DB for production)
users = {}

# Load model
model = load_model('animal_classifier_mobilenetv2.keras')
class_names = ['cats', 'dogs', 'snakes']

# ---------------- Routes ----------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if username in users:
            return "Username already exists!"
        users[username] = password
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect("/")
        else:
            return "Invalid username or password"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect("/login")

@app.route("/", methods=["GET", "POST"])
def index():
    if 'username' not in session:
        return redirect("/login")

    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            predicted_class = class_names[np.argmax(preds)]
            conf = np.max(preds) * 100

            prediction = predicted_class
            confidence = f"{conf:.2f}"
            image_url = url_for('static', filename=f"uploads/{filename}")

    return render_template("index.html", prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
