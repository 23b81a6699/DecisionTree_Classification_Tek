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
# Load Model
# ------------------------------------------------
model = pickle.load(open("models/DecisionTree_Classifier.pkl", "rb"))

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
    Predict whether the mushroom is edible or poisonous
    using Decision Tree Classification 🚀
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
# Encoding Dictionaries
# ------------------------------------------------

cap_shape_dict = {
    "Bell": 0,
    "Conical": 1,
    "Convex": 5,
    "Flat": 2,
    "Knobbed": 3,
    "Sunken": 4
}

cap_surface_dict = {
    "Fibrous": 0,
    "Grooves": 1,
    "Scaly": 3,
    "Smooth": 2
}

cap_color_dict = {
    "Brown": 4,
    "Buff": 0,
    "Cinnamon": 1,
    "Gray": 3,
    "Green": 2,
    "Pink": 5,
    "Purple": 6,
    "Red": 7,
    "White": 8,
    "Yellow": 9
}

bruises_dict = {
    "Bruises Present": 1,
    "No Bruises": 0
}

odor_dict = {
    "Almond": 0,
    "Anise": 1,
    "Creosote": 2,
    "Fishy": 3,
    "Foul": 4,
    "Musty": 5,
    "No Odor": 6,
    "Pungent": 7,
    "Spicy": 8
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
# Prediction
# ------------------------------------------------

if st.button("🎯 Predict Mushroom Type"):

    # Create Input Data

    input_data = np.array([
        [
            cap_shape_dict[cap_shape],
            cap_surface_dict[cap_surface],
            cap_color_dict[cap_color],
            bruises_dict[bruises],
            odor_dict[odor]
        ]
    ])

    # Prediction

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    edible_prob = probability[0][0] * 100

    poisonous_prob = probability[0][1] * 100

    # Result

    if prediction[0] == 1:

        result = "☠️ Poisonous Mushroom"

        message = "⚠️ This mushroom is poisonous. Do NOT consume it."

        confidence = poisonous_prob

    else:

        result = "✅ Edible Mushroom"

        message = "🍽️ This mushroom is edible."

        confidence = edible_prob

    # Display Result

    st.markdown(
        f"""
        <div class='prediction-box'>
            {result}
            <br><br>
            <span>{message}</span>
            <br><br>
            Prediction Confidence : {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------
# Feature Information
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