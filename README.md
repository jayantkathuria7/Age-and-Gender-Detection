# Age and Gender Detection

This project implements an **Age and Gender Detection** model using deep learning. It is built with a Convolutional Neural Network (CNN) trained on the UTKFace dataset, which contains images labeled with age, gender, and ethnicity information. The model predicts the age and gender of a person based on their facial image.

## Key Features:
- **Age Prediction**: Estimates the age of a person from their facial image.
- **Gender Prediction**: Determines the gender (male or female) based on the facial features.
- **Real-time Detection**: Deploys a real-time model for age and gender detection through a user-friendly Streamlit application.

## Project Overview:
- The model was trained using a CNN architecture, which was optimized for the age and gender prediction task.
- A **Streamlit** web app was created for easy interaction, where users can upload images and get predictions in real time.
  
## **Technologies Used**:
- **Python**: The core programming language.
- **TensorFlow**: For building and training the deep learning model.
- **Keras**: A high-level neural networks API used to define the CNN architecture.
- **OpenCV**: For handling image pre-processing and augmentations.
- **Streamlit**: For deploying the real-time web application.
  
## **Kaggle Notebook**:
You can view and run the full notebook, where I implemented the model training and evaluation, on Kaggle:  
[Age and Gender Detection Notebook on Kaggle](https://www.kaggle.com/code/jayantkathuria/age-and-gender-detection)

## **Streamlit App**:
The deployed real-time web application allows users to upload an image and get predictions for age and gender.  
[Age and Gender Detection App](https://age-gender-detection.streamlit.app/)

## **Dataset**:
The model is trained on the **UTKFace dataset**, which contains over 20,000 face images with associated age, gender, and ethnicity labels. You can access the dataset from Kaggle:  
[UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)

## **Setup and Installation**:
1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/jayantkathuria7/Age-and-Gender-Detection.git
    cd Age-and-Gender-Detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. For training the model locally, use the provided Jupyter notebooks in the `notebooks/` directory.

## **Model Training**:
1. Preprocess images from the UTKFace dataset.
2. Train a Convolutional Neural Network (CNN) model on the dataset.
3. Evaluate the model's performance based on accuracy and loss metrics.

## **Future Improvements**:
- Integrate additional features like ethnicity prediction.
- Enhance the model's accuracy by experimenting with other deep learning architectures.
- Incorporate more data for improved generalization.
  
## **Contributing**:
Feel free to fork this repository, create an issue, or submit a pull request if you'd like to contribute to the project.
