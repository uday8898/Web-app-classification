import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    # Open the image file in binary read mode
    with open(image_file, "rb") as f:
        img_data = f.read()  # Read the image data

    # Encode the image data to base64 format
    b64_encoded = base64.b64encode(img_data).decode()

    # Create a CSS style block to set the background image
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    
    # Inject the CSS style into the Streamlit app
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Convert image to 224x224 pixels
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to a numpy array
    image_array = np.asarray(image)

    # Normalize image data to the range [-1, 1]
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Prepare the image data to fit the model input shape
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make a prediction with the model
    prediction = model.predict(data)

    # Determine the index based on the threshold of 0.95
    index = 0 if prediction[0][0] > 0.95 else 1

    # Get the predicted class name and confidence score
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
