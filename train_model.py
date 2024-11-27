import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('./data/alphanorm_anjha.csv')  # Adjust path as needed

# Display the dataset head
print("Dataset head:")
print(df.head())

# Drop unnecessary columns
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)
if 'sex' in df.columns:
    df.drop('sex', axis=1, inplace=True)

# Encode the target variable
le = LabelEncoder()
df['phenotype'] = le.fit_transform(df['phenotype'])

# Split features and target
X = df.drop('phenotype', axis=1)
y = df['phenotype']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(le, encoder_file)

print("Model and label encoder saved successfully!")
