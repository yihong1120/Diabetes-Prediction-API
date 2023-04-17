import requests

def get_predictions(data):
    """
    Send a POST request to the prediction API with the input data and
    return the received predictions.
    
    :param data: A dictionary containing a list of input data for predictions
    :return: A list of predictions, or None if there was an error
    """
    url = 'http://localhost:5000/predict'

    try:
        # Send a POST request with the input data as JSON
        response = requests.post(url, json=data)
        response.raise_for_status()

        # Return the predictions from the response JSON
        return response.json()['predictions']

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    data = {
        'data': [
            [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            [1, 85, 66, 29, 0, 26.6, 0.351, 31],
        ],
    }

    # Call the get_predictions() function with the input data
    predictions = get_predictions(data)

    # Print the predictions if they were successfully received, otherwise print an error message
    if predictions is not None:
        print('Predictions:', predictions)
    else:
        print('Failed to get predictions')
