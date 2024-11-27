from flask import Flask, request, render_template,jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load the trained model and label encoder
model = pickle.load(open('model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input features from the form
    features = [float(x) for x in request.form.values()]
    
    # Define the feature names
    feature_names = ['hb', 'pcv', 'rbc', 'mcv', 'mch', 'mchc', 'rdw', 
                     'wbc', 'neut', 'lymph', 'plt', 'hba', 'hba2', 'hbf']
    
    # Create a DataFrame for the input features
    final_features = pd.DataFrame([features], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(final_features)
    output = label_encoder.inverse_transform(prediction)
    
    return render_template('index.html', prediction_text=f'Predicted Phenotype: {output[0]}')

if __name__ == "__main__":
    app.run(debug=True)
