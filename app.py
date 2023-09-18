# docker build -t mldash-app .
# docker run -p 8081:80 --name mydash mldash-app
# docker pull postgres
# docker run --name mypg -p 5051:5432 -e POSTGRES_PASSWORD=postgres -d postgres


# Import necessary libraries
from flask import Flask, request, render_template
import joblib

# Create a Flask application
app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('model.pkl')

print('Hi 3')

# Define a route for the home page
@app.route('/home')
def home():
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # try:
    # Get user input from the form
    features = [float(request.form['feature1']), float(request.form['feature2'])]

    # Make a prediction using the loaded model
    prediction = model.predict([features])[0]

    # Return the prediction to the user
    return f'Predicted Value: {prediction:.2f}'
    # except Exception as e:
    #     return str(e)

if __name__ == '__main__':
    print('Hi 1')
    app.run(debug=True,host='0.0.0.0', port=80)
    print('Hi 2')