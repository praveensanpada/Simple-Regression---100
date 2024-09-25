from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Load the trained model if it exists
try:
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global model
    # Get the file from the request
    file = request.files['file']
    df = pd.read_csv(file)
    
    # Prepare the data
    X = df.iloc[:, 0:1].values
    y = df.iloc[:, -1].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    with open('linear_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Test the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return jsonify({'message': 'Model trained successfully!', 'mse': mse, 'r2_score': r2})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if not model:
        return jsonify({'error': 'Model not trained yet. Please upload a dataset and train the model first.'})

    # Get the input value from the request
    cgpa = float(request.form['cgpa'])
    
    # Predict the LPA based on the input CGPA
    lpa = model.predict(np.array([[cgpa]]))[0]
    
    return jsonify({'cgpa': cgpa, 'predicted_lpa': lpa})

if __name__ == '__main__':
    app.run(debug=True)
