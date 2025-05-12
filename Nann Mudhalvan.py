from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the model (this should be a pre-trained model file)
model = pickle.load(open('model/model.pkl', 'rb'))
print("Model loaded successfully.")  # Ensure model is loaded

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get the features from the form
            features = [float(request.form[col]) for col in [
                'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
                'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
                'Education', 'Income', 'Diabetes_012', 'HighBP'
            ]]
            
            print("Input features:", features)  # Debugging line to check input values
            
            # Prepare the input features to match the model's trained format
            df_input = pd.DataFrame([features], columns=[
                'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
                'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
                'Education', 'Income', 'Diabetes_012', 'HighBP'
            ])
            
            print("Input DataFrame:", df_input)  # Check the formatted input
            
            # Get prediction
            prediction = model.predict(df_input)[0]
            print("Prediction:", prediction)  # Debugging line to check the prediction
            
            return render_template('index.html', prediction=prediction)
        except Exception as e:
            print(f"Error: {e}")  # Debugging line to check errors
            return render_template('index.html', error="Invalid input, please check your values.")
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
