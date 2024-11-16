import streamlit as st
import numpy as np
from PIL import Image
import os
import pickle  

# Set model directory and path
model_directory = r'D:\Atma\Sem 5\Pembelajaran Mesin\P7\Tugas5_B_11668'
model_path = os.path.join(model_directory, r'best_model.pkl')

# Check if model exists
if os.path.exists(model_path):
    try:
        # Load the model
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        
        class_names = [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        
        def preprocess_image(image):
            image = image.resize((28, 28))  
            image = image.convert('L')  
            image_array = np.array(image)  
            image_array = image_array.reshape(1, -1)  
            return image_array

       
        st.title("Fashion MNIST Image Classifier")
        st.write("Upload some fashion item images (e.g., shoes, bags, shirts), and the model will classify them.")

        
        uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        
        with st.sidebar:
            st.write("## Navigator")
            predict_button = st.button("Predict")

            if uploaded_files and predict_button:
                st.write("### Prediction Results")

                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)  
                    processed_image = preprocess_image(image)  
                    predictions = model.predict_proba(processed_image) 
                    predicted_class = np.argmax(predictions)  
                    confidence = np.max(predictions) * 100 

                    # Display prediction results
                    st.write(f"**File Name:** {uploaded_file.name}")
                    st.write(f"**Predicted Class:** {class_names[predicted_class]}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.write("---")

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image: {uploaded_file.name}", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.error("Model not found")
