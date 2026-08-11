import os
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_data():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "heart_disease.csv")
    return pd.read_csv(file_path)

data = load_data()

# =========================================================
# FEATURES AND TARGET
# =========================================================
X = data.drop(columns=["target_binary"])
y = data["target_binary"]

# =========================================================
# TRAIN MODEL
# =========================================================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    return model, accuracy, cm, report, importance

model, accuracy, cm, report, importance = train_model(X, y)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(135deg, #7f1d1d, #ef4444);
    padding: 34px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin: 0;
}

.hero p {
    font-size: 17px;
    margin-top: 8px;
}

.result-positive {
    background: #fff1f2;
    border: 2px solid #e11d48;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}

.result-negative {
    background: #ecfdf5;
    border: 2px solid #10b981;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}

.result-value {
    font-size: 34px;
    font-weight: 800;
}

.info-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    min-height: 145px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ❤️ Heart Disease")
    st.caption("Machine Learning Prediction System")

    st.divider()

    st.markdown("### Model")
    st.write("Random Forest Classifier")

    st.markdown("### Dataset")
    st.write(f"{len(data):,} records")

    st.markdown("### Input Features")
    st.write(f"{len(X.columns)} health-related features")

    st.divider()
    st.success("Model Status: Ready")

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction</h1>
    <p>Predict the heart disease outcome of a patient using a Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Dataset Records", f"{len(data):,}")
c2.metric("Input Features", len(X.columns))
c3.metric("Model", "Random Forest")
c4.metric("Accuracy", f"{accuracy:.2%}")

st.write("")

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("### 🩺 Enter Patient Details")
st.caption(
    "Enter the values in the same format used by the supplied heart_disease.csv dataset."
)

left, right = st.columns(2)

# Dataset-based defaults
defaults = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
    "num": 0
}

values = {}

# ---------------------------------------------------------
# Left column
# ---------------------------------------------------------
with left:
    values["age"] = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=int(defaults["age"]),
        step=1
    )

    values["sex"] = st.selectbox(
        "Sex",
        [0, 1],
        index=1,
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
    )

    values["cp"] = st.number_input(
        "Chest Pain Type (cp)",
        min_value=0,
        max_value=10,
        value=int(defaults["cp"]),
        step=1
    )

    values["trestbps"] = st.number_input(
        "Resting Blood Pressure (trestbps)",
        min_value=0,
        max_value=300,
        value=int(defaults["trestbps"]),
        step=1
    )

    values["chol"] = st.number_input(
        "Cholesterol (chol)",
        min_value=0,
        max_value=700,
        value=int(defaults["chol"]),
        step=1
    )

    values["fbs"] = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl (fbs)",
        [0, 1],
        index=1,
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

    values["restecg"] = st.number_input(
        "Resting ECG (restecg)",
        min_value=0,
        max_value=5,
        value=int(defaults["restecg"]),
        step=1
    )

# ---------------------------------------------------------
# Right column
# ---------------------------------------------------------
with right:
    values["thalach"] = st.number_input(
        "Maximum Heart Rate (thalach)",
        min_value=0,
        max_value=300,
        value=int(defaults["thalach"]),
        step=1
    )

    values["exang"] = st.selectbox(
        "Exercise-Induced Angina (exang)",
        [0, 1],
        index=0,
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

    values["oldpeak"] = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=20.0,
        value=float(defaults["oldpeak"]),
        step=0.1
    )

    values["slope"] = st.number_input(
        "Slope (slope)",
        min_value=0,
        max_value=5,
        value=int(defaults["slope"]),
        step=1
    )

    values["ca"] = st.number_input(
        "Major Vessels (ca)",
        min_value=0,
        max_value=10,
        value=int(defaults["ca"]),
        step=1
    )

    values["thal"] = st.number_input(
        "Thalassemia (thal)",
        min_value=0.0,
        max_value=10.0,
        value=float(defaults["thal"]),
        step=1.0
    )

    values["num"] = st.number_input(
        "Original Target Score (num)",
        min_value=0,
        max_value=10,
        value=int(defaults["num"]),
        step=1
    )

# =========================================================
# PREDICTION
# =========================================================
st.write("")

if st.button(
    "❤️ Predict Heart Disease",
    type="primary",
    use_container_width=True
):
    patient = pd.DataFrame([values], columns=X.columns)

    prediction = int(model.predict(patient)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(patient)[0]
        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )
        disease_probability = float(class_probabilities.get(1, 0))
        no_disease_probability = float(class_probabilities.get(0, 0))
    else:
        disease_probability = 1.0 if prediction == 1 else 0.0
        no_disease_probability = 1.0 - disease_probability

    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-value">⚠️ HEART DISEASE DETECTED</div>
            <p>Model prediction: class 1</p>
            <p>Disease probability: {disease_probability:.1%}</p>
            <p>No-disease probability: {no_disease_probability:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-value">✅ NO HEART DISEASE PREDICTED</div>
            <p>Model prediction: class 0</p>
            <p>No-disease probability: {no_disease_probability:.1%}</p>
            <p>Disease probability: {disease_probability:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

    st.warning(
        "This is a machine-learning demonstration and is not a medical diagnosis."
    )

# =========================================================
# MODEL INFORMATION
# =========================================================
st.write("")
st.markdown("### 🤖 Model Information")

a, b = st.columns(2)

with a:
    st.markdown("""
    <div class="info-card">
    <h4>🌲 Random Forest Classifier</h4>
    <b>Trees:</b> 100<br>
    <b>Train/Test Split:</b> 80% / 20%<br>
    <b>Random State:</b> 42<br>
    <b>Target:</b> target_binary
    </div>
    """, unsafe_allow_html=True)

with b:
    st.markdown("""
    <div class="info-card">
    <h4>📊 Evaluation</h4>
    The model is evaluated using accuracy, confusion matrix,
    classification report, and feature importance.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
st.write("")
with st.expander("📊 View Feature Importance"):
    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# DATASET
# =========================================================
with st.expander("📁 View Training Dataset"):
    st.dataframe(
        data,
        use_container_width=True,
        height=350
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================
with st.expander("📈 View Model Evaluation"):
    st.write("Confusion Matrix")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.write("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

st.divider()

st.caption(
    "Heart Disease Prediction • Python • Pandas • Scikit-learn • Streamlit • Random Forest"
)