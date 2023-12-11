import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# st.set_option('deprecation.showfileUploaderEncoding', False)

model = tf.keras.models.load_model('D:\\PCD\\UAS\\MobileNetPCD.h5')

def apply_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    return binary_image

def preprocess_image(image):
    # Ensure the input image has three channels
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = tf.cast(image, tf.float32)
    image /= 255.0
    # Resize image to match the model's expected input shape
    image = tf.image.resize(image, [224, 224])
    image = np.expand_dims(image, axis=0)
    return image

def predict_class(image, model):
    prediction = model.predict(image)[0]
    class_names = ['Strawberry Healthy', 'Strawberry Leaf Scorch']
    
    max_prob_index = np.argmax(prediction)
    
    predicted_class = class_names[max_prob_index]
    confidence = prediction[max_prob_index] * 100
    
    return predicted_class, confidence


#STREAMLIT APP
st.title('Strawberry Classifier')
file = st.file_uploader("Upload an image of strawberry leaf", type=["jpg", "png"])

if file is None:
    st.text('Waiting for upload....')
else:
    slot = st.empty()
    slot.text('Running inference....')

    test_image = Image.open(file)

    st.image(test_image, caption="Input Image", width=400)

    # Apply thresholding to the image
    thresholded_image = apply_threshold(np.asarray(test_image))
    thresholded_image_display = cv2.resize(thresholded_image, (224, 224))
    st.image(thresholded_image_display, caption="Thresholded Image", width=400)

    # Use the thresholded image for prediction
    thresholded_image = preprocess_image(thresholded_image)  # preprocess the thresholded image
    pred_class, confidence = predict_class(thresholded_image, model)
    output = f'The prediction is: {confidence:.2f}% {pred_class}'
    slot.text('Done')
    st.success(output)