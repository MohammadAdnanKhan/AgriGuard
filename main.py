import streamlit as st
import tensorflow as tf
import numpy as np
import time

class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def model_prediction(test_image):
    model = tf.keras.models.load_model("Trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Converting single image to batch
    predictions = model.predict(input_arr)[0]  # Getting prediction for single image
    top3_indices = predictions.argsort()[-3:][::-1]  # Top-3 predictions indices i.e. Indices of top 3 probabilities
    top3_predictions = [(class_name[i], predictions[i]) for i in top3_indices]  # Get labels and scores
    return top3_predictions, top3_indices[0]  # Return top-3 and highest confidence index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.markdown("<h1 style='text-align: center;'>AgriGuard 🌿</h1>", unsafe_allow_html=True)
    image_path = "Plant disease image.jpg"
    st.image(image_path, use_column_width=True)
    
    st.markdown("""
        # Welcome to AgriGuard 🌱🌾

        **AgriGuard** is your intelligent plant health assistant, designed to efficiently identify plant diseases. Upload a photo of a plant leaf, and our system will analyze it using advanced machine learning to detect signs of potential diseases. Let’s work together to protect our crops and secure healthier harvests! 🌍🍃

        ## 🌿 How AgriGuard Works
        1. **Upload Image** 📸: Navigate to the **Disease Recognition** page and upload a clear image of a plant leaf showing possible symptoms.
        2. **Advanced Analysis** 🧠: AgriGuard will process the image using state-of-the-art algorithms to detect any disease indicators.
        3. **Instant Results** 📝: View a ranked list of the top 3 potential disease categories, along with their confidence scores.

        ## 🌟 Why Choose AgriGuard?
        - **High Accuracy** 🎯: Powered by advanced machine learning models trained on a diverse dataset for accurate disease detection.
        - **User-Friendly Interface** 💻: Designed with simplicity in mind for a smooth and intuitive experience.
        - **Fast and Efficient** ⚡: Get results in seconds, enabling quick and informed decision-making.

        ## 🚀 Get Started
        Begin your plant health journey by clicking on **Disease Recognition** in the sidebar to upload an image. Let AgriGuard take care of the rest!

        ## ℹ️ About Us
        Discover more about the dataset and the creator behind AgriGuard on the **About** page. Together, let’s grow healthier plants and contribute to sustainable agriculture! 🌱💚
    """)


# About Page
elif app_mode == "About":
    st.markdown("<h1 style='text-align: center;'>About</h1>", unsafe_allow_html=True)

    # About section content
    st.markdown("""
    ### 🌱 About Dataset
    This dataset was augmented offline based on the original dataset.
    It consists of approximately 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes.
    The total dataset is divided into an 80/20 ratio for training and validation, preserving the directory structure.
    A separate directory containing 33 test images was created later for prediction purposes.
    
    [Click here for the dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

    ### 📊 Content
    - **Train**: 70,295 images
    - **Validation**: 17,572 images
    - **Test**: 33 images

    ### 👤 About Author
    - **Name**: Mohd Adnan Khan
    - **Background**: An enthusiastic data scientist and machine learning engineer with a strong passion for applying technology to solve real-world problems. Skilled in various domains, from text and image classification to predictive analytics. Driven by a curiosity for cutting-edge innovations, I am dedicated to leveraging AI to make processes more efficient and accessible across industries.
    - [LinkedIn](https://www.linkedin.com/in/mohd-adnan--khan)
    - [GitHub](https://github.com/MohammadAdnanKhan)
    - [Kaggle](https://www.kaggle.com/mohdadnankhan1)
    - **Contact**: mohdadnankhan.india@gmail.com

    ### 🔮 Future Improvements
    - Expanding the dataset to include more plant species
    - Developing mobile and offline versions of the app
    - Adding additional disease severity insights and treatment recommendations

    ### 💡 Project Motivation
    - AgriGuard was born from a desire to leverage machine learning to support sustainable agriculture by providing farmers with fast, reliable plant disease detection.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.markdown("<h1 style='text-align: center;'>Disease Recognition 🌿</h1>", unsafe_allow_html=True)

    test_image = st.file_uploader("Upload an Image of a Plant Leaf 🍃:", type=["jpg", "jpeg", "png"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict 🔍"):
        if test_image is not None:
            start_time = time.time() 

            # Spinner for better user experience
            with st.spinner("Analyzing the image...🧠"):
                time.sleep(1)
                st.success("Prediction complete! ✅")

            # Make prediction
            top3_predictions, result_index = model_prediction(test_image)
            end_time = time.time()

            # Display Prediction and Confidence Score
            st.subheader("Prediction Result")
            st.write(f"**Predicted Disease**: {top3_predictions[0][0]}")
            st.write(f"**Confidence**: {top3_predictions[0][1] * 100:.2f}%")
            st.write(f"**Time taken for prediction**: {end_time - start_time:.2f} seconds")

            # Display Top-3 Predictions
            st.subheader("Top 3 Predicted Diseases")
            for label, confidence in top3_predictions:
                st.write(f"- {label}: {confidence * 100:.2f}% confidence")

            # Download Report
            report = f"Prediction: {top3_predictions[0][0]}\nConfidence: {top3_predictions[0][1] * 100:.2f}%\n"
            st.download_button("Download Prediction Report 📥", report, file_name="prediction_report.txt", mime="text/plain")

        else:
            st.error("Please upload an image before making a prediction. ❌")
            
# Sidebar Information
st.sidebar.subheader("About AgriGuard 🌿")
st.sidebar.text("Upload a plant leaf image to detect diseases.")
st.sidebar.text("Our system provides the top 3 possible diseases with confidence scores.")
st.sidebar.markdown("Visit **Disease Recognition** to start.")
st.sidebar.markdown("Learn more on the **About** page.")
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset")
st.sidebar.markdown("[Click here for dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)")
st.sidebar.markdown("---")
st.sidebar.subheader("Contact")
st.sidebar.markdown("+91 6306941902")
st.sidebar.markdown("📧 [mohdadnankhan.india@gmail.com](mailto:mohdadnankhan.india@gmail.com)")
st.sidebar.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/mohd-adnan--khan)")