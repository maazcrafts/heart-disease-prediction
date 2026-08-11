❤️ Heart Disease Prediction

A Machine Learning Web Application that predicts heart disease using a Random Forest Classifier.

📌 Project Overview

This project uses the heart_disease.csv dataset and trains a Random Forest classification model to predict the binary heart disease outcome.

The target column used by the program is:

target_binary

🛠️ Technologies Used

Python

Pandas

Scikit-learn

Streamlit

Random Forest Classifier

🤖 Machine Learning Algorithm

Random Forest Classifier

The model is created with:

RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

The dataset is divided into training and testing data using an 80/20 split with stratification.

📊 Dataset

The application loads:

heart_disease.csv

The target column is removed from the input features:

X = data.drop(columns=["target_binary"])
y = data["target_binary"]

🔢 Input Features

The web application uses the feature columns available in the supplied dataset, including the following heart-related attributes:

Age

Sex

Chest Pain Type (cp)

Resting Blood Pressure (trestbps)

Cholesterol (chol)

Fasting Blood Sugar (fbs)

Resting ECG (restecg)

Maximum Heart Rate (thalach)

Exercise-Induced Angina (exang)

ST Depression (oldpeak)

Slope (slope)

Major Vessels (ca)

Thalassemia (thal)

num

🔄 Project Workflow

Heart Disease Dataset
        ↓
Load Dataset
        ↓
Separate Features and Target
        ↓
Train/Test Split
        ↓
Random Forest Training
        ↓
Model Evaluation
        ↓
Enter Patient Details
        ↓
Prediction
        ↓
Display Result

📈 Model Evaluation

The original Python program evaluates the trained model using:

Accuracy Score

Confusion Matrix

Classification Report

Feature Importance

The Streamlit interface provides these evaluation details through expandable sections.

🌐 Web Interface

The Streamlit application provides:

❤️ Heart Disease Prediction dashboard

Patient input form

Random Forest model information

Model accuracy

Heart disease prediction

Prediction probabilities

Feature importance

Confusion matrix

Classification report

Training dataset viewer

🎯 Prediction Output

The system predicts one of the binary outcomes:

Heart Disease Detected

or

No Heart Disease Predicted

The web interface also displays the model's class probabilities.

📁 Project Structure

Heart-Disease-Prediction/
│
├── heart-disease-prediction.py
├── heart_disease.csv
├── requirements-heart-disease.txt
└── README.md

💻 Installation

Install the required libraries:

pip install -r requirements-heart-disease.txt

Or install them manually:

pip install streamlit pandas scikit-learn

▶️ Run the Web Application

Make sure the following files are in the same folder:

heart-disease-prediction.py
heart_disease.csv
requirements-heart-disease.txt

Then run:

streamlit run heart-disease-prediction.py

⚠️ Disclaimer

This project is intended for educational and machine-learning demonstration purposes. The prediction is not a medical diagnosis and should not be used as a substitute for professional medical advice.

👨‍💻 Author

Maaz Khan
