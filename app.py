from flask import Flask, render_template, request

app = Flask(__name__)

import pandas as pd
import json
import ast
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data and train the model (you can reuse your existing code)
data = pd.read_csv('https://raw.githubusercontent.com/VarunPalrecha/DataSets/main/output1.csv')

X = data[['positiveReviews']].values
data['sales'] = data[['sales_month1', 'sales_month2', 'sales_month3']].sum(axis=1)
y = data['sales'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    new_positive_reviews = float(request.form.get('positive_reviews'))
    
    if new_positive_reviews < 0:
        error_message = "Error: The number of positive reviews can't be a negative number."
        return render_template('index.html', error_message=error_message)
    
    new_positive_reviews = [[new_positive_reviews]]
    predicted_sales = model.predict(new_positive_reviews)[0]
    
    return render_template('index.html', predicted_sales=predicted_sales)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

