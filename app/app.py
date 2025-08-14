import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load CNN model
cnn_model = load_model('/home/silver/Pneumonia-Detection-using-CNN/models/cnn_model.keras')

# Streamlit UI
st.title("Chest X-Ray Pneumonia Detection")


# Display sample image under the title
sample_image = Image.open("/home/silver/Pneumonia-Detection-using-CNN/images/iStock-2079858839.jpg")
st.image(sample_image, caption="Sample Chest X-ray", use_container_width=True)

# Description
st.markdown("""
**Upload one or more X-Ray images**, and this app will predict whether each image shows **Pneumonia** or **Normal**.  
This model uses **CNN** to make the predictions.
""")

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        # Convert image to grayscale
        if image.mode == "RGB":
            image = image.convert("L")
        
        # Resize to 150x150 for CNN
        image = image.resize((150, 150))
        
        # Normalize pixel values
        img_array = np.array(image) / 255.0
        
        # Reshape for CNN
        img_cnn = img_array.reshape(1, 150, 150, 1)
        return img_cnn
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

# Multi-file upload
uploaded_files = st.file_uploader(
    "Upload one or more X-ray Images", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    images = []
    for file in uploaded_files:
        try:
            img = Image.open(file)
            images.append(img)
        except Exception as e:
            st.error(f"Error loading file {file.name}: {e}")

    # Preview uploaded images
    st.subheader("Preview Images")
    cols = st.columns(min(len(images), 4))  # Up to 4 images per row
    for i, img in enumerate(images):
        with cols[i % len(cols)]:
            st.image(img, caption=f"Image {i+1}", use_container_width=True)

    # Button to run predictions
    if st.button("Classify Images"):
        st.subheader("Prediction Results")
        for i, img in enumerate(images):
            st.write(f"Classifying Image {i+1}...")
            img_cnn = preprocess_image(img)
            if img_cnn is not None:
                cnn_preds = cnn_model.predict(img_cnn)
                cnn_label = "Pneumonia" if cnn_preds[0][0] > 0.5 else "Normal"
                st.write(f"- CNN Prediction: **{cnn_label}**")
            else:
                st.write("- CNN Prediction: Error in processing")
else:
    st.info("Please upload one or more images to classify.")
