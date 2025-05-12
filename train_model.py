import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Create 'model' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load the dataset
df = pd.read_excel("Naan mudhalvan Dataset (1).xlsx")
df.dropna(inplace=True)

# Define features and target
X = df.drop(columns=['HeartDiseaseorAttack'])  # Drop the target variable from the input features
y = df['HeartDiseaseorAttack']

# Convert categorical variables if needed
X = pd.get_dummies(X)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to model/model.pkl
pickle.dump(model, open("model/model.pkl", "wb"))

print("✅ Model trained and saved to 'model/model.pkl'")
