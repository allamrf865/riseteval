import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Research Scientist Evaluation",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #F8F9FA, #E9ECEF);
            font-family: 'Poppins', sans-serif;
        }
        .header {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .header h1 {
            font-size: 2.5rem;
            color: #007BFF;
            text-align: center;
        }
        .header p {
            font-size: 1.2rem;
            color: #495057;
            text-align: center;
        }
        .dataset-card {
            border: 1px solid #DEE2E6;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
            background: white;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }
        .dataset-card h4 {
            margin: 0;
            color: #007BFF;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>📊 AI Research Scientist Evaluation</h1>
        <p>Analyze up to 10 datasets with advanced metrics and a sleek, interactive interface.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.title("📂 Upload Your Datasets")
uploaded_files = st.sidebar.file_uploader(
    "Upload up to 10 Excel files",
    type=["xlsx"],
    accept_multiple_files=True,
    help="You can upload multiple Excel files (up to 10)."
)

# Function to calculate and display metrics
def evaluate_metrics(y_true, y_pred):
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    sensitivitas = tp / (tp + fn) if (tp + fn) != 0 else 0
    spesifisitas = tn / (tn + fp) if (tn + fp) != 0 else 0
    nilai_duga_positif = tp / (tp + fp) if (tp + fp) != 0 else 0
    nilai_duga_negatif = tn / (tn + fn) if (tn + fn) != 0 else 0
    prevalensi = (tp + fn) / len(y_true) if len(y_true) > 0 else 0
    rasio_kemungkinan_positif = sensitivitas / (1 - spesifisitas) if (1 - spesifisitas) != 0 else np.inf
    rasio_kemungkinan_negatif = (1 - sensitivitas) / spesifisitas if spesifisitas != 0 else np.inf
    akurasi = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    
    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_value = auc(fpr, tpr)
    
    return {
        "Sensitivitas (Recall)": sensitivitas,
        "Spesifisitas": spesifisitas,
        "Nilai Duga Positif": nilai_duga_positif,
        "Nilai Duga Negatif": nilai_duga_negatif,
        "Prevalensi": prevalensi,
        "Rasio Kemungkinan Positif": rasio_kemungkinan_positif,
        "Rasio Kemungkinan Negatif": rasio_kemungkinan_negatif,
        "Akurasi": akurasi,
        "AUC": auc_value
    }, fpr, tpr, auc_value

# Main layout for datasets
if uploaded_files:
    st.markdown("### 📊 Datasets Overview")
    if len(uploaded_files) > 10:
        st.error("You can only upload a maximum of 10 datasets!")
    else:
        tabs = st.tabs([f"Dataset {i+1}" for i in range(len(uploaded_files))])

        for i, uploaded_file in enumerate(uploaded_files):
            with tabs[i]:
                # Load dataset
                try:
                    df = pd.read_excel(uploaded_file)
                    st.markdown(f"#### Dataset {i + 1}: {uploaded_file.name}")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")

                # Allow selection of target column and positive class
                target_col = st.selectbox(
                    f"Select target column for Dataset {i + 1}",
                    options=df.columns,
                    key=f"target_col_{i}"
                )
                pos_label = st.selectbox(
                    f"Select positive class for Dataset {i + 1}",
                    options=df[target_col].unique(),
                    key=f"pos_label_{i}"
                )

                # Analyze dataset button
                if st.button(f"Analyze Dataset {i + 1}", key=f"analyze_{i}"):
                    with st.spinner("Analyzing data..."):
                        # Data preprocessing
                        df[target_col] = df[target_col].apply(lambda x: 1 if x == pos_label else 0)
                        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
                        y = df[target_col]

                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                        # Train model
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)

                        # Make predictions
                        y_pred = model.predict(X_test)
                        y_pred_prob = model.predict_proba(X_test)[:, 1]

                        # Calculate metrics
                        metrics, fpr, tpr, auc_value = evaluate_metrics(y_test, y_pred)

                        # Display metrics
                        st.markdown("#### Model Metrics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("AUC", f"{auc_value:.2f}")
                        col2.metric("Sensitivity", f"{metrics['Sensitivitas (Recall)']:.2f}")
                        col3.metric("Specificity", f"{metrics['Spesifisitas']:.2f}")
                        col4, col5 = st.columns(2)
                        col4.metric("Accuracy", f"{metrics['Akurasi']:.2f}")

                        # Display all other metrics
                        st.markdown("#### Additional Metrics")
                        for metric, value in metrics.items():
                            if metric not in ["AUC", "Sensitivity", "Specificity", "Accuracy"]:
                                st.metric(metric, f"{value:.2f}")

                        # Confusion Matrix
                        st.markdown("#### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
                        plt.title("Confusion Matrix")
                        plt.xlabel("Predicted")
                        plt.ylabel("Actual")
                        st.pyplot(fig)

                        # ROC Curve
                        st.markdown("#### ROC Curve")
                        fig, ax = plt.subplots()
                        plt.plot(fpr, tpr, label=f"AUC = {auc_value:.2f}")
                        plt.plot([0, 1], [0, 1], 'r--', label="Random Guess")
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.title("ROC Curve")
                        plt.legend()
                        st.pyplot(fig)
else:
    st.markdown("### Please upload datasets to get started!")
