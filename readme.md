# AgriGuard 🌿

**AgriGuard** is an intelligent plant health assistant designed to help identify plant diseases with high accuracy. This application allows users to upload an image of a plant leaf, analyze it using a CNN model, and receive predictions about potential diseases. AgriGuard is ideal for farmers, agronomists, and researchers aiming to ensure plant health and improve agricultural productivity.

## 📋 Table of Contents
- 📌 [Project Overview](#project-overview)
- ✨ [Features](#features)
- 🌐 [Demo](#demo)
- ⚙️ [Installation](#installation)
- 🚀 [Usage](#usage)
- 📊 [Dataset](#dataset)
- 🧠 [Model Details](#model-details)
- 🔮 [Future Improvements](#future-improvements)
- 👤 [Author](#author)

---

## 📌 Project Overview
AgriGuard leverages a Convolutional Neural Network (CNN) model trained on a dataset of plant leaf images. The model can classify leaves into 38 categories, identifying both the type of plant and the specific disease (if any). AgriGuard’s user-friendly interface, built with Streamlit, provides an intuitive experience for users to upload images and quickly receive prediction results.

## ✨ Features
- **🦠 Disease Detection**: Upload an image of a plant leaf, and AgriGuard predicts if it has any disease, along with the specific disease type.
- **📈 Top-3 Predictions**: The system provides the top-3 likely diseases for each image, along with confidence scores.
- **💻 Streamlit Interface**: A fast and responsive user interface built with Streamlit.
- **📝 Downloadable Report**: After predictions, users can download a text report of the results.

## 🌐 Link to AgriGuard
Check out AgriGuard on Streamlit: [AgriGuard](https://agriguard.streamlit.app/).

## ⚙️ Installation
To run AgriGuard locally, follow these steps:

1. **📂 Clone the Repository**:
    ```bash
    git clone https://github.com/MohammadAdnanKhan/AgriGuard.git
    cd AgriGuard
    ```

2. **📦 Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3. **⬇️ Download the Model**:
   Place your trained model file (`Trained_model.keras`) in the project directory.

4. **🚀 Run the Application**:
    ```bash
    streamlit run app.py
    ```

## 🚀 Usage
1. Launch the application using `streamlit run app.py`.
2. Navigate to the "Disease Recognition" page.
3. Upload a plant leaf image (formats supported: `.jpg`, `.jpeg`, `.png`).
4. Click on **Predict 🔍** to see the results.
5. Download the prediction report if needed.

## 📊 Dataset
The dataset used to train this model consists of 87,000 images of healthy and diseased leaves, divided into 38 classes. The dataset can be accessed from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

- **📚 Train**: 70,295 images
- **📊 Validation**: 17,572 images
- **🔬 Test**: 33 images

## 🧠 Model Details
AgriGuard uses a CNN model trained on a large dataset of plant leaves, allowing it to identify various diseases with high accuracy. The model predicts the disease and displays the top-3 results with confidence scores.

## 🔮 Future Improvements
- **🌱 Expand Plant and Disease Database**: Add more plant species and diseases to increase the scope and versatility of AgriGuard.
- **🌍 Multi-Language Support**: Enable multiple languages to make AgriGuard more accessible to a global audience, particularly for non-English speaking users.
- **🔍 Model Interpretability**: Integrate visual tools (e.g., saliency maps) to show which areas of the image influenced the model's decision, improving transparency and user trust.
- **📈 Disease Severity Insights and Treatment Recommendations**: Provide insights into disease severity and suggest treatments, helping users make informed decisions about plant care.

## 👤 Author
**Mohd Adnan Khan**  
Data Scientist and Machine Learning Engineer passionate about applying AI to solve real-world problems.

- 💼 [LinkedIn](https://www.linkedin.com/in/mohd-adnan--khan)
- 🐙 [GitHub](https://github.com/MohammadAdnanKhan)
- 📊 [Kaggle](https://www.kaggle.com/mohdadnankhan1)
- **📧 Contact**: mohdadnankhan.india@gmail.com