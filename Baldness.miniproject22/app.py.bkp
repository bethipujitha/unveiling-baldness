from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and encoded data
model = pickle.load(open('model.pkl', 'rb'))
encoded = pickle.load(open('encoded_data.pkl', 'rb'))

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
            input_features = [float(x) for x in request.form.values()]
            x = np.array(input_features).reshape(1, -1)
            data = pd.DataFrame(x)
            print(data)
            pred = model.predict(data)
            print(pred)

            # Decode the prediction using the encoded labels
            label = encoded.inverse_transform(pred)
            print(label)

            if label[0] == 'No Hairfall':
                return render_template('output.html', predict="Patient Has No Hairfall")
            else:
                return render_template('output.html', predict="Patient Has Hairfall")
        except Exception as e:
            return render_template('error.html', error=str(e))
    else:
        return render_template('output.html')

if __name__ == "__main__":
    app.run(port=4000, debug=False)
