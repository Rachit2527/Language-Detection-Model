import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('Language Detection.csv')

# Extract features and labels
x = data['Text']
y = data['Language']

# Vectorize text data
cv = CountVectorizer()
x = cv.fit_transform(x)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
reg = LogisticRegression()
reg.fit(x_train, y_train)

# Streamlit app
st.title("Language Detection App")

# Text input for user
user_text = st.text_area("Enter text:", "Your text goes here")

# Button to predict
if st.button("Detect Language"):
    # Transform user input
    user_data = cv.transform([user_text]).toarray()

    # Make prediction
    prediction = reg.predict(user_data)

    # Display prediction
    st.success("Predicted Language:")
    st.write(prediction[0])
