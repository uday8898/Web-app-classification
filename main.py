import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from util import classify, set_background

# Set the background image for the Streamlit app
set_background('./bgs/bg5.png')

# Set the title of the app
st.title('Pneumonia classification')

# Set a header to prompt users to upload an image
st.header('Please upload a chest X-ray image')

# Create a file uploader widget allowing users to upload image files
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load the pre-trained model for pneumonia classification
model = load_model('model/pneumonia_classifier.h5')

# Load the class names from the 'labels.txt' file
with open('model/labels.txt', 'r') as f:
    # Read all lines, strip newline characters, split each line, and extract class names
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    # Close the file (not strictly necessary within the 'with' block)
    f.close()

# Check if a file has been uploaded
if file is not None:
    # Open the uploaded file as an image and convert it to RGB mode
    image = Image.open(file).convert('RGB')
    
    # Display the uploaded image in the Streamlit app
    st.image(image, use_column_width=True)
    
    # Classify the uploaded image using the pre-trained model and class names
    class_name, conf_score = classify(image, model, class_names)
    
    # Display the predicted class name
    st.write("## {}".format(class_name))
    
    # Display the confidence score of the classification, formatted as a percentage
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
