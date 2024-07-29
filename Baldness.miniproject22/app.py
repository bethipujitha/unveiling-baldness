from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model and encoded data
try:
    model = pickle.load(open('model.pkl', 'rb'))
    encoded = pickle.load(open('encoded_data.pkl', 'rb'))
except Exception as e:
    app.logger.error(f"Error loading model or encoded data: {e}")
    raise e
label_mapping = {0: 'No Hairfall', 1: 'Hairfall'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inner-page')
def inner():
    return render_template('inner-page.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        try:
            # Collect input features from the form
            input_features = []
            for key, value in request.form.items():
                app.logger.debug(f"Received input {key}: {value}")
                if value.strip() == '':
                    raise ValueError(f"Input for {key} is empty.")
                try:
                    input_features.append(float(value))
                except ValueError as ve:
                    raise ValueError(f"Invalid input for {key}: {value} (must be a number).") from ve

            app.logger.debug(f"Input features: {input_features}")

            if len(input_features) != 12:
                raise ValueError("Expected 12 input features.")

            x = np.array(input_features[:2]).reshape(1, 2)  # Explicitly reshape to (1, 12)
            data = pd.DataFrame(x)
            app.logger.debug(f"Data for prediction: {data}")

            pred = model.predict(data)
            app.logger.debug(f"Model prediction: {pred}")
            # app.logger.debug(f"{encoded}")

            # Decode the prediction using the encoded labels
            #label = encoded.inverse_transform(pred)
            #app.logger.debug(f"Decoded prediction: {label}")
            decoded_label = label_mapping.get(int(pred[0]), 'Unknown')  
            if decoded_label == 'No Hairfall':
                return render_template('output.html', predict="Patient Has No Hairfall")
            else:
                return render_template('output.html', predict="Patient Has Hairfall")
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            return render_template('error.html', error=str(e))
    else:
        return render_template('output.html')

if __name__ == "__main__":
    app.run(port=4000, debug=False)
