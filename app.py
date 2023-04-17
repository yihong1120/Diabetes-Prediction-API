from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model from file
with open('models/model_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive input data as JSON, make predictions using the loaded model,
    and return the predictions as JSON.
    """
    try:
        # Get the request data
        data = request.json

        # Validate the input data format
        if 'data' not in data or not isinstance(data['data'], list):
            raise ValueError("Invalid input data format")

        # Convert the input data to a NumPy array
        new_data = np.array(data['data'])

        # Make predictions using the loaded model
        new_data_predictions = model.predict(new_data)

        # Return the predictions as JSON
        return jsonify(predictions=new_data_predictions.tolist())

    except ValueError as e:
        return jsonify(error=str(e)), 400

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
