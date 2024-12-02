from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import os
from lib import flood_detection  # Import flood detection function from lib.py

# Flask app setup
app = Flask(__name__, static_folder="assets", template_folder="templates")
CORS(app)

# Hardcoded city-to-image mapping
city_image_map = {
    "new_york": "new_york[1].png",
    "new_york_alt": "new_york[2].png",
    "new_orleans": "new_orleans[1].png",
    "new_orleans_alt": "new_orleans[2].png",
    "little_rock": "little_rock[1].png",
    "little_rock_alt": "little_rock[2].png",
    "irvine": "irvine[1].png",
    "irvine_alt": "irvine[2].png"
}


@app.route("/")
def home():
    """Serve the home page."""
    return render_template("final.html")

@app.route("/predict", methods=['POST'])
def predict():
    """
    Handle flood prediction requests.
    Expects JSON with "city_name".
    """
    data = request.json
    city_name = data.get("city_name", "").lower()  # Convert to lowercase for consistency

    if not city_name:
        return jsonify({"error": "City name is required."}), 400

    # Get the image path for the requested city
    image_path = city_image_map.get(city_name)
    if not image_path:
        return jsonify({"error": f"No data found for {city_name}."}), 404

    # Construct the full image path
    image_path_full = os.path.join(os.getcwd(), image_path)

    try:
        # Perform flood detection
        model_path = "RGB_CNN.h5"  # Path to the RGB model
        prediction = flood_detection("rgb", model_path, image_path_full)

        # Return the prediction and image path
        return jsonify({
            "city": city_name.capitalize(),
            "prediction": prediction,
            "image_path": f"/{image_path}"  # Relative path for frontend usage
        })
    except Exception as e:
        return jsonify({"error": f"Error processing the image for {city_name}: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

