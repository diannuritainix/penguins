import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.title("🐧 Penguin Species Classifier")
st.write("A simple machine learning web app using Streamlit & Seaborn Penguins Dataset.")

# Load dataset
penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()

# Encode categorical columns
le_species = LabelEncoder()
penguins["species_encoded"] = le_species.fit_transform(penguins["species"])

le_island = LabelEncoder()
penguins["island_encoded"] = le_island.fit_transform(penguins["island"])

le_sex = LabelEncoder()
penguins["sex_encoded"] = le_sex.fit_transform(penguins["sex"])

penguins.drop(['species','island','sex'],axis=1, inplace=True)

# Features & target
X = penguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", 
              "island_encoded", "sex_encoded"]]
y = penguins["species_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

st.sidebar.header("Input Features")
bill_length = st.sidebar.slider("Bill Length (mm)", 32.0, 60.0, 43.0)
bill_depth = st.sidebar.slider("Bill Depth (mm)", 13.0, 22.0, 17.0)
flipper_length = st.sidebar.slider("Flipper Length (mm)", 170, 240, 200)
body_mass = st.sidebar.slider("Body Mass (g)", 2500, 6500, 4200)

island = st.sidebar.selectbox("Island", le_island.classes_)
sex = st.sidebar.selectbox("Sex", le_sex.classes_)

# Convert to numeric encoded values
input_data = pd.DataFrame({
    "bill_length_mm": [bill_length],
    "bill_depth_mm": [bill_depth],
    "flipper_length_mm": [flipper_length],
    "body_mass_g": [body_mass],
    "island_encoded": [le_island.transform([island])[0]],
    "sex_encoded": [le_sex.transform([sex])[0]]
})

# Predict
prediction = model.predict(input_data)[0]
prediction_label = le_species.inverse_transform([prediction])[0]
prob = model.predict_proba(input_data).max()

st.subheader("🔮 Prediction Result")
st.write(f"**Predicted Species:** {prediction_label}")
st.write(f"**Confidence:** {prob:.2f}")


st.header("📊 Data Visualization")

# ====== 1. Correlation Heatmap ======
st.subheader("🔎 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(penguins.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ====== 2. Scatter Plot Interaktif ======
st.subheader("🟣 Scatter Plot Explorer")

num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

x_axis = st.selectbox("Select X Axis", num_cols)
y_axis = st.selectbox("Select Y Axis", num_cols, index=1)

fig2, ax2 = plt.subplots()
sns.scatterplot(
    data=penguins,
    x=x_axis,
    y=y_axis,
    hue="species_encoded",
    palette="Set2"
)
st.pyplot(fig2)

# ====== 3. Distribution Plot ======
st.subheader("📦 Feature Distribution")

feature = st.selectbox("Select Feature to View Distribution", num_cols)

fig3, ax3 = plt.subplots()
sns.histplot(penguins[feature], kde=True)
st.pyplot(fig3)

st.write("---")
st.write("Built with Streamlit + Random Forest + Penguins Dataset 🐧")
