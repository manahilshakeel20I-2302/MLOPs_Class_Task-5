from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')

# Iris dataset feature names for the web form
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    data = [float(request.form[feature]) for feature in feature_names]

    # Convert input data to NumPy array and reshape
    data = np.array([data])

    # Predict using the loaded model
    prediction = model.predict(data)

    # Map the prediction to the iris target names
    iris_types = ['Setosa', 'Versicolor', 'Virginica']
    predicted_class = iris_types[prediction[0]]

    return render_template('index.html', prediction_text=f'The predicted iris species is: {predicted_class}')

if __name__ == '__main__':
    app.run(debug=True)
