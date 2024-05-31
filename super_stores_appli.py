import streamlit as st
import joblib
import pandas as pd

# Loading saved models
model_regr = joblib.load('model_super_store_regr.pkl')
model_svm = joblib.load('model_super_store_svm.pkl')
model_rf = joblib.load('model_super_store_rf.pkl')

# Definition of the function for making predictions
def predict_sales(model, X):
    prediction = model.predict(X)
    return prediction

# Configuring the Streamlit application
st.set_page_config(page_title="Sales Prediction")
st.title("Prediction of the number of delivery days after ordering")
st.image('White.png')
# Sidebar for selecting the model
st.sidebar.title("Select the model")
model_choice = st.sidebar.selectbox("Choose a model", ["Linear Regression", "SVM", "Random Forest"])

# Form for entering data
st.sidebar.title("Enter data")
sales = st.sidebar.number_input("Sales", min_value=0.0, step=0.01)
order_day = st.sidebar.number_input("Order Day", min_value=1, max_value=31, step=1)
ship_day = st.sidebar.number_input("Delivery Day", min_value=1, max_value=31, step=1)
order_year = st.sidebar.number_input("Order Year", min_value=2000, max_value=2100, step=1)
ship_year = st.sidebar.number_input("Delivery Year", min_value=2000, max_value=2100, step=1)

# Button to make the prediction
if st.sidebar.button("Predict"):
    # Creating a DataFrame with the entered data
    data = pd.DataFrame({
        'Sales': [sales],
        'order_day': [order_day],
        'ship_day': [ship_day],
        'order_year': [order_year],
        'ship_year': [ship_year]
    })
    
    # Making the prediction with the selected model
    if model_choice == "Linear Regression":
        prediction = predict_sales(model_regr, data)
    elif model_choice == "SVM":
        prediction = predict_sales(model_svm, data)
    else:
        prediction = predict_sales(model_rf, data)
    
    # Displaying the prediction result
        #st.image('White.png')
    st.success(f"The number of delivery days is: {prediction[0]}")
