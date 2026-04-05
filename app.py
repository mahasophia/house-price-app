import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("🏠 House Price Prediction (Multiple Linear Regression)")

# Load dataset
df = pd.read_csv("housing.csv")

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Check missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Description
st.subheader("Statistical Summary")
st.write(df.describe())

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Features & Target
X = df[['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'NOX', 'AGE']]
y = df['MEDV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
st.subheader("Model Evaluation")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"MSE: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Coefficients
st.subheader("Feature Importance (Coefficients)")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
st.write(coef_df)

# Plot actual vs predicted
st.subheader("Actual vs Predicted Prices")
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred)
ax2.set_xlabel("Actual Prices")
ax2.set_ylabel("Predicted Prices")
st.pyplot(fig2)

# User input prediction
st.subheader("Predict House Price")

rm = st.number_input("RM (rooms)", value=6.0)
lstat = st.number_input("LSTAT (%)", value=10.0)
ptratio = st.number_input("PTRATIO", value=18.0)
indus = st.number_input("INDUS", value=10.0)
nox = st.number_input("NOX", value=0.5)
age = st.number_input("AGE", value=50.0)

if st.button("Predict"):
    input_data = np.array([[rm, lstat, ptratio, indus, nox, age]])
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: {prediction[0]:.2f}")
