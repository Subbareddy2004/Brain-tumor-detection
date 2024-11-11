import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .css-1v0mbdj.etr89bj1 {
        width: 100%;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained hybrid CNN-ViT model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('brain_tumor_classification_model.h5')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# Mapping the predicted indices to tumor names with colors
tumor_info = {
    0: {'name': 'Glioma', 'color': '#FF6B6B', 'description': 'A tumor that starts in the glial cells of the brain'},
    1: {'name': 'Meningioma', 'color': '#4ECDC4', 'description': 'A tumor that forms in the meninges'},
    2: {'name': 'Pituitary', 'color': '#45B7D1', 'description': 'A tumor that develops in the pituitary gland'},
    3: {'name': 'No Tumor', 'color': '#96CEB4', 'description': 'No tumor detected in the scan'}
}

def preprocess_image(image, img_size=(224, 224)):
    img = np.array(image.convert('L'))
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def draw_bounding_box(image, is_tumor_present, prediction_class):
    img = np.array(image)
    if is_tumor_present:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Convert hex color to BGR
            color = tuple(int(tumor_info[prediction_class]['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            color = color[::-1]  # Reverse for BGR
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    return img

def classify_tumor(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][predicted_class_index]) * 100
    return predicted_class_index, confidence

# Header with custom styling
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50; padding: 1.5rem;'>
        🧠 Brain Tumor Detection And Classification Using CNN And ViT
    </h1>
""", unsafe_allow_html=True)

# Information box
st.info("""
    Welcome to the Brain Tumor Classification System! This application uses advanced AI to analyze 
    brain MRI scans and detect various types of tumors. Simply upload an MRI image to get started.
""")

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    # File uploader with custom styling
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

with col2:
    if uploaded_file is not None:
        # Classify and display results
        with st.spinner("Analyzing MRI scan..."):
            predicted_class_index, confidence = classify_tumor(image)
            tumor_type = tumor_info[predicted_class_index]['name']
            is_tumor_present = tumor_type != 'No Tumor'

            # Display results in a colored box
            result_color = tumor_info[predicted_class_index]['color']
            st.markdown(f"""
                <div style='background-color: {result_color}; padding: 1rem; border-radius: 0.5rem; color: white;'>
                    <h3>Analysis Results</h3>
                    <p><strong>Classification:</strong> {tumor_type}</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                    <p><strong>Description:</strong> {tumor_info[predicted_class_index]['description']}</p>
                </div>
            """, unsafe_allow_html=True)

            # Display processed image with bounding box
            if is_tumor_present:
                processed_image = draw_bounding_box(image, is_tumor_present, predicted_class_index)
                processed_image = Image.fromarray(processed_image)
                st.image(processed_image, caption="Detected Region", use_column_width=True)

# Display tumor type information
st.markdown("### Types of Brain Tumors")
cols = st.columns(len(tumor_info))
for idx, (col, (_, info)) in enumerate(zip(cols, tumor_info.items())):
    with col:
        st.markdown(f"""
            <div style='background-color: {info['color']}; padding: 1rem; border-radius: 0.5rem; color: white;'>
                <h4>{info['name']}</h4>
                <p>{info['description']}</p>
            </div>
        """, unsafe_allow_html=True)

# # Footer
# st.markdown("""
#     <div style='text-align: center; padding: 2rem;'>
#         <p style='color: #666;'>Developed with ❤️ for medical imaging analysis</p>
#     </div>
# """, unsafe_allow_html=True)