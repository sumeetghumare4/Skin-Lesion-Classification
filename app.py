import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained models
model1 = load_model('model.h5')
# model2 = load_model('model2.h5')
# model3 = load_model('model3.h5')

# Define the classes
class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

# Streamlit app layout
st.title('Skin Cancer Image Classifier')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    # Resize the image to the required input size of the models
    resized_image = image.resize((180, 180))
    # Convert the image to a numpy array
    img_array = np.array(resized_image)
    # Normalize the image
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Placeholder for classification result
    result = None

    # Choose a model to classify the image
    # For simplicity, let's use model1 for now
    model = model1

    # Perform image classification
if st.button('Predict'):
    prediction = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)
    # Get the class label
    result = class_names[predicted_class_index]

    # Display the classification result
    st.write('Prediction:', result)
