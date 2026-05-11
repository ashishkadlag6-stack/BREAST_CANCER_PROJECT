import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Breast Cancer Classification", page_icon="🩺", layout="wide")

# Title and description
st.title("🩺 Breast Cancer Classification with Boosting Algorithms")
st.markdown("""
This app demonstrates the use of AdaBoost and Gradient Boosting classifiers
on the Breast Cancer Wisconsin dataset to predict malignant or benign tumors.
""")

# Sidebar for controls
st.sidebar.header("Model Configuration")
n_estimators = st.sidebar.slider("Number of Estimators", 50, 200, 100, 10)
learning_rate = st.sidebar.slider("Learning Rate (Gradient Boosting)", 0.01, 0.5, 0.1, 0.01)
max_depth = st.sidebar.slider("Max Depth (Gradient Boosting)", 1, 5, 3, 1)

if st.button("🚀 Train Models"):
    with st.spinner("Training models... This may take a moment."):

        # 1. Load Cancer Dataset
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names

        # Display dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", f"{X.shape[0]} samples")
        with col2:
            st.metric("Features", f"{X.shape[1]} features")
        with col3:
            st.metric("Classes", f"{len(target_names)} ({', '.join(target_names)})")

        # 2. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Preprocessing: Scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 4. Apply Boosting Algorithms

        # AdaBoost Classifier
        st.subheader("🤖 AdaBoost Classifier")
        ada_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators,
            random_state=42
        )
        ada_model.fit(X_train_scaled, y_train)
        y_pred_ada = ada_model.predict(X_test_scaled)
        accuracy_ada = accuracy_score(y_test, y_pred_ada)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("AdaBoost Accuracy", f"{accuracy_ada:.4f}")
        with col2:
            st.metric("Training Samples", len(X_train))

        # Gradient Boosting Classifier
        st.subheader("🌳 Gradient Boosting Classifier")
        grad_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        grad_model.fit(X_train_scaled, y_train)
        y_pred_grad = grad_model.predict(X_test_scaled)
        accuracy_grad = accuracy_score(y_test, y_pred_grad)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gradient Boosting Accuracy", f"{accuracy_grad:.4f}")
        with col2:
            st.metric("Test Samples", len(X_test))

        # 5. Take Best Result and Select Model
        st.subheader("🏆 Model Comparison")
        if accuracy_ada > accuracy_grad:
            best_model = ada_model
            best_accuracy = accuracy_ada
            y_pred_best = y_pred_ada
            model_name = "AdaBoost Classifier"
        else:
            best_model = grad_model
            best_accuracy = accuracy_grad
            y_pred_best = y_pred_grad
            model_name = "Gradient Boosting Classifier"

        st.success(f"**Best Model Selected:** {model_name} with **{best_accuracy:.4f}** accuracy!")

        # 6. Evaluate Matrix and Print Accuracy
        st.subheader(f"📊 Evaluation for {model_name}")

        # Classification Report
        st.text("Classification Report:")
        report = classification_report(y_test, y_pred_best, target_names=target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4))

        # Confusion Matrix Visualization
        cm = confusion_matrix(y_test, y_pred_best)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_title(f'Confusion Matrix for {model_name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

        # Store models and scaler in session state for prediction
        st.session_state['best_model'] = best_model
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = feature_names
        st.session_state['target_names'] = target_names
        st.session_state['model_name'] = model_name

# Prediction section
st.header("🔮 Make Predictions")
if 'best_model' in st.session_state:
    st.write(f"Using the trained {st.session_state['model_name']}")

    # Option to use sample data or manual input
    prediction_mode = st.radio("Choose input method:", ["Use sample data", "Manual input"])

    if prediction_mode == "Use sample data":
        # Load data again to get test samples
        data = load_breast_cancer()
        X_test = data.data
        sample_idx = st.slider("Select sample index", 0, len(X_test)-1, 0)

        sample_data = X_test[sample_idx:sample_idx+1]
        st.write("Sample features:")
        sample_df = pd.DataFrame(sample_data, columns=st.session_state['feature_names'])
        st.dataframe(sample_df)

    else:
        # Manual input
        st.write("Enter feature values:")
        cols = st.columns(3)
        features = {}
        feature_names = st.session_state['feature_names']

        for i, feature in enumerate(feature_names):
            col_idx = i % 3
            with cols[col_idx]:
                features[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"feature_{i}"
                )

        sample_data = np.array([list(features.values())])

    if st.button("Predict"):
        # Scale the data
        sample_scaled = st.session_state['scaler'].transform(sample_data)

        # Make prediction
        prediction = st.session_state['best_model'].predict(sample_scaled)
        prediction_proba = st.session_state['best_model'].predict_proba(sample_scaled)

        result = st.session_state['target_names'][prediction[0]]
        confidence = prediction_proba[0][prediction[0]]

        if result == 'malignant':
            st.error(f"⚠️ Prediction: **{result.upper()}** (Confidence: {confidence:.4f})")
        else:
            st.success(f"✅ Prediction: **{result.upper()}** (Confidence: {confidence:.4f})")

else:
    st.info("👆 Please train the models first by clicking the 'Train Models' button above.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and scikit-learn")