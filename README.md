# Diabetes Prediction API

This repository contains a RESTful API implementation for predicting diabetes in Pima Indians based on various diagnostic measurements. The API is built using Flask and XGBoost. The project includes training and saving the best model as well as testing the API using Python requests.

## Prerequisites

Ensure that you have the following installed on your machine:

- Python 3.6 or later
- pandas
- scikit-learn
- numpy
- Flask
- requests
- xgboost

## Dataset

The dataset contains information about female patients at least 21 years old of Pima Indian heritage. The task is to predict whether a patient has diabetes based on various diagnostic measurements.

Dataset source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

## Project Structure

* **'train_and_save_xgb_model.py'**: Trains an XGBoost classifier on the Pima Indians Diabetes dataset, selects the best model using grid search with cross-validation, and saves it to a file.
* **'app.py'**: Flask application file that loads the saved model and provides an API endpoint for making predictions.
* **'test_api.py'**: Python script to test the API by sending a POST request with input data and printing the received predictions.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yihong1120/Diabetes-Prediction-Api.git
cd Diabetes-Prediction-Api
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Train and save the XGBoost model:

```bash
python train_and_save_xgb_model.py
```

4. Run the Flask application:

```bash
python app.py
```

5. Test the API with sample input data:

```bash
python test_api.py
```

## API Documentation

### POST '/predict'

Endpoint for making predictions using the saved XGBoost model.

Request
* JSON payload containing an array of input data

```json
{
  "data": [
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],
    [1, 85, 66, 29, 0, 26.6, 0.351, 31]
  ]
}
```

Response
* JSON object containing an array of predictions
```json
{
  "predictions": [1, 0]
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/yihong1120/Diabetes-Prediction-Api/blob/main/LICENSE) file for details.
