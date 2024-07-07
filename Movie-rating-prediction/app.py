#THIS FILE SHOWS HOW THE MODEL IS USED WITH FLASK APP.
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

data = pd.read_csv("./IMDb Movies India.csv", encoding='latin-1')
app = Flask(__name__)

# Load the pre-trained model
filename = 'linear_regression_model.sav'
loaded_model = joblib.load(filename)

# Define the function to predict movie rating
def predict_rating(Year, Votes, Duration, Genre_mean_rating, Director_encoded, Actor1_encoded, Actor2_encoded, Actor3_encoded):
    features = np.array([Year, Votes, Duration, Genre_mean_rating, Director_encoded, Actor1_encoded, Actor2_encoded, Actor3_encoded]).reshape(1, -1)
    prediction = loaded_model.predict(features)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/data')
def show_data():
    sampled_data = pd.DataFrame()

    while sampled_data.empty:
        sampled_data = data.sample(n=10)
        if sampled_data.isnull().values.any():
            sampled_data = pd.DataFrame()  

    # Convert the sampled data to HTML
    data_head = sampled_data.to_html(classes='data', index=False)
    return render_template('data.html', data_head=data_head)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        Year = int(request.form['Year'])
        Votes = float(request.form['Votes'])
        Duration = int(request.form['Duration'])
        Genre = request.form['Genre']
        Director = request.form['Director']
        Actor1 = request.form['Actor1']
        Actor2 = request.form['Actor2']
        Actor3 = request.form['Actor3']

        # Encode Genre, Director, and Actors based on mean ratings
        Genre_mean_rating = data[data['Genre'] == Genre]['Rating'].mean()
        Director_encoded = data[data['Director'] == Director]['Rating'].mean()
        Actor1_encoded = data[data['Actor 1'] == Actor1]['Rating'].mean()
        Actor2_encoded = data[data['Actor 2'] == Actor2]['Rating'].mean()
        Actor3_encoded = data[data['Actor 3'] == Actor3]['Rating'].mean()

        # Predict using the loaded model
        prediction = predict_rating(Year, Votes, Duration, Genre_mean_rating, Director_encoded, Actor1_encoded, Actor2_encoded, Actor3_encoded)

        # Render the predicted result and sampled data
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
