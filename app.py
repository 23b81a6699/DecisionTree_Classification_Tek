import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

# ------------------------------------------------
# Load CSS
# ------------------------------------------------
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------------------------
# Load Model & Encoders
# ------------------------------------------------
model = pickle.load(open("models/DecisionTree_Classifier.pkl", "rb"))

encoders = pickle.load(open("models/label_encoders.pkl", "rb"))

# ------------------------------------------------
# Title
# ------------------------------------------------
st.markdown(
    "<h1 class='main-title'>🍄 Mushroom Classification App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='info-box'>
    Predict whether the mushroom is edible or poisonous using
    Decision Tree Classification with GridSearchCV 🚀
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.header("⚙️ Model Information")

st.sidebar.success("Algorithm : Decision Tree Classifier")

st.sidebar.success("Hyperparameter Tuning : GridSearchCV")

st.sidebar.success("Dataset : Mushroom Classification")

# ------------------------------------------------
# Feature Dictionaries
# ------------------------------------------------

cap_shape_dict = {
    "Bell": "b",
    "Conical": "c",
    "Convex": "x",
    "Flat": "f",
    "Knobbed": "k",
    "Sunken": "s"
}

cap_surface_dict = {
    "Fibrous": "f",
    "Grooves": "g",
    "Scaly": "y",
    "Smooth": "s"
}

cap_color_dict = {
    "Brown": "n",
    "Buff": "b",
    "Cinnamon": "c",
    "Gray": "g",
    "Green": "r",
    "Pink": "p",
    "Purple": "u",
    "Red": "e",
    "White": "w",
    "Yellow": "y"
}

bruises_dict = {
    "Bruises Present": "t",
    "No Bruises": "f"
}

odor_dict = {
    "Almond": "a",
    "Anise": "l",
    "Creosote": "c",
    "Fishy": "y",
    "Foul": "f",
    "Musty": "m",
    "No Odor": "n",
    "Pungent": "p",
    "Spicy": "s"
}

# ------------------------------------------------
# User Inputs
# ------------------------------------------------

st.subheader("🧾 Enter Mushroom Features")

col1, col2, col3 = st.columns(3)

with col1:

    cap_shape = st.selectbox(
        "Cap Shape",
        list(cap_shape_dict.keys())
    )

    cap_surface = st.selectbox(
        "Cap Surface",
        list(cap_surface_dict.keys())
    )

with col2:

    cap_color = st.selectbox(
        "Cap Color",
        list(cap_color_dict.keys())
    )

    bruises = st.selectbox(
        "Bruises",
        list(bruises_dict.keys())
    )

with col3:

    odor = st.selectbox(
        "Odor",
        list(odor_dict.keys())
    )

# ------------------------------------------------
# Convert Inputs to Dataset Codes
# ------------------------------------------------

cap_shape_value = cap_shape_dict[cap_shape]

cap_surface_value = cap_surface_dict[cap_surface]

cap_color_value = cap_color_dict[cap_color]

bruises_value = bruises_dict[bruises]

odor_value = odor_dict[odor]

# ------------------------------------------------
# Prediction
# ------------------------------------------------

if st.button("🎯 Predict Mushroom Type"):

    # Create Input DataFrame

    input_df = pd.DataFrame({
        "cap-shape": [cap_shape_value],
        "cap-surface": [cap_surface_value],
        "cap-color": [cap_color_value],
        "bruises": [bruises_value],
        "odor": [odor_value]
    })

    # --------------------------------------------
    # Label Encoding
    # --------------------------------------------

    for col in input_df.columns:

        value = input_df[col].iloc[0]

        encoded_value = encoders[col].transform([value])[0]

        input_df[col] = encoded_value

    # --------------------------------------------
    # Prediction
    # --------------------------------------------

    prediction = model.predict(input_df)

    prediction_proba = model.predict_proba(input_df)

    edible_prob = prediction_proba[0][0] * 100

    poisonous_prob = prediction_proba[0][1] * 100

    # --------------------------------------------
    # Display Result
    # --------------------------------------------

    if prediction[0] == 1:

        result = "☠️ Poisonous Mushroom"

        message = "⚠️ This mushroom is poisonous. Do NOT consume it."

        probability = poisonous_prob

    else:

        result = "✅ Edible Mushroom"

        message = "🍽️ This mushroom is edible."

        probability = edible_prob

    st.markdown(
        f"""
        <div class='prediction-box'>
            {result}
            <br><br>
            <span>{message}</span>
            <br><br>
            Prediction Confidence : {probability:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------
# Feature Information Section
# ------------------------------------------------

st.markdown("---")

st.subheader("📖 Feature Meanings")

with st.expander("Cap Shape"):
    st.write("""
    • Bell  
    • Conical  
    • Convex  
    • Flat  
    • Knobbed  
    • Sunken
    """)

with st.expander("Cap Surface"):
    st.write("""
    • Fibrous  
    • Grooves  
    • Scaly  
    • Smooth
    """)

with st.expander("Cap Color"):
    st.write("""
    • Brown  
    • Buff  
    • Cinnamon  
    • Gray  
    • Green  
    • Pink  
    • Purple  
    • Red  
    • White  
    • Yellow
    """)

with st.expander("Bruises"):
    st.write("""
    • Bruises Present  
    • No Bruises
    """)

with st.expander("Odor"):
    st.write("""
    • Almond  
    • Anise  
    • Creosote  
    • Fishy  
    • Foul  
    • Musty  
    • No Odor  
    • Pungent  
    • Spicy
    """)

# ------------------------------------------------
# Footer
# ------------------------------------------------

st.markdown(
    """
    <hr>
    <center>
    <h4>✨ Built using Streamlit, Scikit-Learn & GridSearchCV ✨</h4>
    </center>
    """,
    unsafe_allow_html=True
)