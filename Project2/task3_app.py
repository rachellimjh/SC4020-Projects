import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch 
import torch.nn as nn 
import joblib
import math 

class MLP(nn.Module):
    
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
         
            )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def load_all_assets():
    try:
        # read cancer_data to get bounds of the features later on as well
        cancer_data = pd.read_csv("datasets/cancer_data.csv")
        # cancer pred assets
        cancer_model = MLP(30) # 30 features
        cancer_model.load_state_dict(torch.load('task3_models/cancer_model.pth'))
        cancer_model.eval() 
        cancer_scaler = joblib.load('task3_models/cancer_scaler.joblib')

        # symptom checker assets
        symptom_model = joblib.load('task3_models/disease_classifier.pkl')
        disease_encoder = joblib.load('task3_models/disease_encoder.joblib')
        symptom_list = pickle.load(open('task3_models/all_symptoms.pkl', 'rb')) 
        symptom_weights_df = joblib.load('task3_models/symptom_weights.joblib')
        URGENT_DISEASES = pickle.load(open('task3_models/urgent_diseases.pkl', 'rb')) 
        
        return cancer_data, cancer_model, cancer_scaler, symptom_model, disease_encoder, symptom_list, symptom_weights_df, URGENT_DISEASES
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load models or assets. Please check your 'models/' folder. Error: {e}")
        st.stop() 

# Load everything into variables
cancer_data, cancer_model, cancer_scaler, symptom_model, disease_encoder, symptom_list, symptom_weights_df, URGENT_DISEASES = load_all_assets()


def predict_cancer_urgency(input_features, model, scaler):
    """Predicts malignancy and classifies urgency using the PyTorch model."""
    
    # 1. Scaling: Use the loaded Joblib scaler`
    features_array = np.array(input_features).reshape(1, -1)
    scaled_data = scaler.transform(features_array)
    
    # 2. Conversion: Convert numpy array to PyTorch Tensor (float type)
    input_tensor = torch.from_numpy(scaled_data).float()
    
    # 3. Prediction: Get the raw logit output
    with torch.no_grad(): 
        # model_output contains logits (unscaled scores)
        logits = model(input_tensor) 
        
        # 4. Apply Sigmoid to convert logits to a probability (0 to 1)
        probability_tensor = torch.sigmoid(logits)
        
        # Convert tensor result to a single Python float
        probability_malignant = probability_tensor.item() 
    
    # 5. Apply Urgency Logic
    if probability_malignant >= 0.75:
        urgency = 'High'
    elif probability_malignant >= 0.30:
        urgency = 'Middle'
    else:
        urgency = 'Low'
    
    return urgency, probability_malignant



URGENCY_THRESHOLD_LOW = 5
URGENCY_THRESHOLD_HIGH = 15

def classify_symptom_urgency(selected_symptoms, model, encoder, symptom_list, weights, urgent_list):
    # initialise empty df with all symptoms absent
    input_symptoms = pd.DataFrame(columns=symptom_list)
    input_symptoms.loc[0] = 0

    for i in selected_symptoms:
        if i in symptom_list:
            input_symptoms.loc[0,i] = 1

    # predict disease using classifier(model)
    input = input_symptoms.values
    probs = model.predict_proba(input)[0]
    predicted_index = np.argmax(probs)
  
    predicted_disease = encoder.inverse_transform([predicted_index])[0]

    # calc total symptom weights
    total_score = 0
 
    for i in selected_symptoms:
        try:
            weight = weights.loc[i].iloc[0] 
            total_score += weight
        except Exception:
            pass 
    
  
    if total_score >= URGENCY_THRESHOLD_HIGH:
        score_urgency = 'High'
    elif total_score >= URGENCY_THRESHOLD_LOW:
        score_urgency = 'Middle'
    else:
        score_urgency = 'Low'
    
    # if listed as urgent then return urgent
    if predicted_disease in urgent_list:
        final_urgency = 'High'
    else:
        final_urgency = score_urgency
        
    return final_urgency, predicted_disease, total_score

# --- 2. Main UI Structure and Navigation ---

st.title("🏥 General Healthcare Prediction and Urgency System")

# The sidebar selection widget for navigation
app_mode = st.sidebar.selectbox(
    "Choose the Assessment Module",
    ("Select Module...", "Symptom-Based Urgency Checker", "Breast Cancer Urgency Prediction")
)

# 3. CONTENT DISPLAY LOGIC (The "Pages")

FEATURE_GROUPS = {
    "Mean Features (Clinical Measurement)": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
        "smoothness_mean", "compactness_mean", "concavity_mean", 
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
    ],
    "SE Features (Standard Error)": [
        "radius_se", "texture_se", "perimeter_se", "area_se", 
        "smoothness_se", "compactness_se", "concavity_se", 
        "concave points_se", "symmetry_se", "fractal_dimension_se"
    ],
    "Worst Features (Largest/Worst Values)": [
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", 
        "smoothness_worst", "compactness_worst", "concavity_worst", 
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
}

if app_mode == "Select Module...":
    st.info("👈 Please select an assessment module from the sidebar to begin.")




elif app_mode == "Breast Cancer Urgency Prediction":
    st.header("🔬 Breast Cancer Urgency Prediction")
    st.markdown("Input the 30 features from the diagnostic image analysis, organized by measurement type.")

    input_features_dict = {} # Use a dictionary to store inputs safely by name

    # --- 1. Define Tabs ---
    tab_mean, tab_se, tab_worst = st.tabs(list(FEATURE_GROUPS.keys()))
    

    ordered_feature_names = sum(FEATURE_GROUPS.values(), [])
    
    # --- 2. Create Sliders within Tabs ---
    feature_bounds = {}
    for col in cancer_data.columns:
        if col != 'diagnosis':
            col_min = cancer_data[col].min()
            col_max = cancer_data[col].max()
            feature_bounds[col] = (col_min, col_max)
    
    def get_bounds(feature_name):
        # default values incase
        default_min, default_max, default_value, step = 0.0, 1.0, 0.5, 0.01
    
        if feature_name in feature_bounds:
            f_min, f_max = feature_bounds[feature_name]
            f_min = float(math.floor(f_min.min()))
            f_max = float(math.ceil(f_max.max()))
            step = (f_max - f_min) / 100.0 
            default_value = (f_min + f_max) / 2.0
            return f_min, f_max, default_value, step
        else:
            return default_min, default_max, default_value, step

    for tab, (group_name, features) in zip([tab_mean, tab_se, tab_worst], FEATURE_GROUPS.items()):
        with tab:
            st.subheader(group_name)
            
            cols = st.columns(2)
            
            for i, feature in enumerate(features):
                # Determine which column to place the slider in
                col = cols[i % 2]
                
                min_v, max_v, default_v, step_v = get_bounds(feature)
                
                # Create the slider and store its value in the dictionary
                input_features_dict[feature] = col.slider(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=min_v, 
                    max_value=max_v, 
                    value=default_v, 
                    step=step_v,
                    key=f"slider_{feature}"
                )

    # --- 3. Prepare Ordered Input Array ---
    
    # After collecting all inputs, reorder them into a list based on the model's required order
    input_features = [input_features_dict[name] for name in ordered_feature_names]

    # --- 4. Prediction Button and Logic ---

    if st.button("Analyze and Predict Urgency"):
   
        if len(input_features) != 30:
             st.error("Internal Error: Could not collect all 30 features.")
             st.stop()
        
 
        urgency, probability_malignant = predict_cancer_urgency(input_features, cancer_model, cancer_scaler)
        
        st.subheader("✅ Prediction Results")
        
        # 5. Apply Urgency Logic
        if urgency == 'High':
            st.error(f"🚨 **HIGH URGENCY**: Malignancy Probability is **{probability_malignant:.2%}**.")
            st.warning("Immediate medical attention is required.")
        elif urgency == 'Middle':
            st.warning(f"⚠️ **MIDDLE URGENCY**: Malignancy Probability is **{probability_malignant:.2%}**.")
            st.info("Consult a specialist within 24-48 hours.")
        else:
            st.success(f"✅ **LOW URGENCY**: Malignancy Probability is **{probability_malignant:.2%}**.")
            st.info("Routine follow-up is advised.")

# --- SYMPTOM CHECKER PAGE ---
elif app_mode == "Symptom-Based Urgency Checker":
    st.header("🩺 Symptom-Based Urgency Checker")
    st.markdown("Select all applicable patient symptoms.")
    
    # This uses your loaded list of all symptoms 
    selected_symptoms = st.multiselect("Select symptoms:", symptom_list)
    
    # Prediction Button
    if st.button("Classify Urgency"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom to proceed.")
            st.stop() # Stop execution if no symptoms are selected

        # 1. CALL THE PREDICTION FUNCTION
        # The variables (S_model, S_encoder, S_list, S_weights_df, URGENT_DISEASES) 
        # were loaded globally at the start of app.py
        urgency, disease, score = classify_symptom_urgency(
            selected_symptoms, symptom_model, disease_encoder, symptom_list, symptom_weights_df, URGENT_DISEASES
        )
        
        # 2. DISPLAY RESULTS
        st.subheader("✅ Assessment Results")
        
        # Display Results based on Urgency level
        if urgency == 'High':
            st.error(f"🚨 **HIGH URGENCY**: Predicted Disease: **{disease}** (Score: {score:.1f}).")
            st.warning("Immediate medical attention is required. Please proceed to the nearest Emergency Room.")
        elif urgency == 'Middle':
            st.warning(f"⚠️ **MIDDLE URGENCY**: Predicted Disease: **{disease}** (Score: {score:.1f}).")
            st.info("Consult a General Practitioner or Urgent Care facility within 1-2 days.")
        else:
            st.success(f"✅ **LOW URGENCY**: Predicted Disease: **{disease}** (Score: {score:.1f}).")
            st.info("Self-care and routine monitoring are recommended. Consult a doctor if symptoms persist.")