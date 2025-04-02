from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("mnist_model.keras")

# Initialize Flask app
app = Flask(__name__)

# Route to render the HTML page
@app.route("/")
def home():
    return render_template("index.html")


def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)  # Convert to numpy array

    # Check if background is white (invert if necessary)
    if np.mean(image) > 127:
        image = 255 - image

    image = image / 255.0  # Normalize to [0,1]
    image = image.reshape(1, 784)  # Flatten the image
    return image


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(file)
        image = preprocess_image(image)

        prediction = model.predict(image)[0]  # Extract probabilities for all digits
        predicted_class = np.argmax(prediction)  # Get the most likely digit

        # Create a dictionary of probabilities for each digit (0-9)
        probabilities = {str(i): f"{prob * 100:.2f}%" for i, prob in enumerate(prediction)}

        return jsonify({
            "predicted_digit": int(predicted_class),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
