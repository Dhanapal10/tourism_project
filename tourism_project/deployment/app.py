
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -----------------------------
# Load trained model
# -----------------------------
model_path = hf_hub_download(
    repo_id="dhanapalpalanisamy/tourism-model",
    filename="model.pkl"
)

model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction")
st.markdown("Predict whether a customer will purchase the Tourism Package.")

# -----------------------------
# LabelEncoder mappings (from tourism.csv)
# -----------------------------
TYPE_OF_CONTACT = {
    "Company Invited": 0,
    "Self Inquiry": 1
}

GENDER = {
    "Female": 0,
    "Male": 1
}

MARITAL_STATUS = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}

OCCUPATION = {
    "Free Lancer": 0,
    "Large Business": 1,
    "Salaried": 2,
    "Small Business": 3
}

PRODUCT_PITCHED = {
    "Basic": 0,
    "Deluxe": 1,
    "King": 2,
    "Standard": 3,
    "Super Deluxe": 4
}

DESIGNATION = {
    "AVP": 0,
    "Director": 1,
    "Executive": 2,
    "Manager": 3,
    "Senior Manager": 4,
    "VP": 5
}

# -----------------------------
# Numeric Inputs
# -----------------------------
age = st.number_input("Age", 18, 70, 35)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration = st.number_input("Duration of Pitch (minutes)", 5, 40, 18)
persons = st.number_input("Number of Persons Visiting", 1, 5, 2)
followups = st.number_input("Number of Followups", 0, 6, 3)
property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
trips = st.number_input("Number of Trips (per year)", 0, 10, 4)
passport = st.selectbox("Has Passport", ["No", "Yes"])
pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Owns a Car", ["No", "Yes"])
children = st.number_input("Number of Children Visiting", 0, 3, 0)
income = st.number_input("Monthly Income", 10000, 200000, 60000)

# -----------------------------
# Categorical Inputs
# -----------------------------
contact = st.selectbox("Type of Contact", list(TYPE_OF_CONTACT.keys()))
occupation = st.selectbox("Occupation", list(OCCUPATION.keys()))
gender = st.selectbox("Gender", list(GENDER.keys()))
product = st.selectbox("Product Pitched", list(PRODUCT_PITCHED.keys()))
marital = st.selectbox("Marital Status", list(MARITAL_STATUS.keys()))
designation = st.selectbox("Designation", list(DESIGNATION.keys()))

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = {
        "Age": age,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups,
        "PreferredPropertyStar": property_star,
        "NumberOfTrips": trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": children,
        "MonthlyIncome": income,
        "TypeofContact": TYPE_OF_CONTACT[contact],
        "Occupation": OCCUPATION[occupation],
        "Gender": GENDER[gender],
        "ProductPitched": PRODUCT_PITCHED[product],
        "MaritalStatus": MARITAL_STATUS[marital],
        "Designation": DESIGNATION[designation]
    }

    df = pd.DataFrame([input_data])

    df = df[model.feature_names_in_]

    prediction = model.predict(df)[0]

    if prediction == 1:
        st.success("Customer is likely to purchase the Tourism Package")
    else:
        st.warning("Customer is unlikely to purchase the Tourism Package")
