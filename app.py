import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import tempfile
from sklearn.cluster import KMeans
import io

# Load your custom model (ensure your model path is correct)
model = tf.keras.models.load_model('Age_Sex_Detection.keras')

def preprocess_image(img):
    if isinstance(img, Image.Image):  # Check if the image is a PIL image
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = np.array(img)  # Convert PIL image to NumPy array
    elif isinstance(img, np.ndarray):  # If already a NumPy array
        if len(img.shape) == 3 and img.shape[2] == 4:  # Check if it's RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)    # Convert RGBA to RGB
        img = np.array(img)
    else:
        raise ValueError("Unsupported image format")

        # Resize and normalize image
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    return img


# Assuming your model outputs two separate predictions: age and gender
def detect_age_gender(image):
    img = preprocess_image(image)
    pred = model.predict(np.array([img]))
    print(pred)
    # Extract predictions
    age = int(np.round(pred[1][0]))
    gender = int(np.round(pred[0][0]))

    # Example interpretation (you need to adjust according to your model's output)
    gender_f = ['Male', 'Female']

    return age, gender_f[gender]

# Detect faces in the frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

# Create shirt region of interest (ROI)
def create_shirt_roi(frame, face_rect):
    x, y, w, h = face_rect
    shirt_width = w * 2
    shirt_height = h * 2
    shirt_x = x - int(shirt_width / 4)
    shirt_y = y + h + int(h / 4)

    shirt_x = max(shirt_x, 0)
    shirt_y = max(shirt_y, 0)
    shirt_width = min(shirt_width, frame.shape[1] - shirt_x)
    shirt_height = min(shirt_height, frame.shape[0] - shirt_y)

    shirt_roi = (shirt_x, shirt_y, shirt_width, shirt_height)
    return shirt_roi

# Detect shirt color
def detect_shirt(frame, shirt_roi):
    shirt_region = frame[shirt_roi[1]:shirt_roi[1] + shirt_roi[3], shirt_roi[0]:shirt_roi[0] + shirt_roi[2]]
    rgb_shirt = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2RGB)
    pixels = rgb_shirt.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    color_histogram = np.bincount(labels, minlength=2)
    dominant_cluster = np.argmax(color_histogram)
    dominant_color = colors[dominant_cluster]
    white_threshold = np.array([200, 200, 200])
    black_threshold = np.array([50, 50, 50])
    if np.all(dominant_color > white_threshold):
        detected_color = 'white'
    elif np.all(dominant_color < black_threshold):
        detected_color = 'black'
    else:
        detected_color = 'unknown'
    return detected_color

def main():
    st.title("Age and Gender Detection")
    st.markdown("""
    **Key Points:**
    - The system detects the number of males and females.
    - It estimates age, with specific adjustments:
      - **White Shirt**: Age set to 23.
      - **Black Shirt**: Categorized as a child.
    - The feature only operates if there are at least 2 people in the room.
    """)
    # Upload image
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file:
        # Load and display the image
        image = Image.open(image_file)
        # display_size = (150, 130)  # Desired size (width, height)
        # image_resized = image.resize(display_size)
        #
        # st.image(image_resized, caption="Uploaded Image")
        image_np = np.array(image)
        # Detect faces in the image
        faces = detect_faces(image_np)
        male_count = 0
        female_count = 0
        if len(faces) >= 2:
            # Draw bounding boxes around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 255), 2)
                face_roi = image_np[y:y + h, x:x + w]
                age, gender = detect_age_gender(face_roi)
                shirt_roi = create_shirt_roi(image_np, (x, y, w, h))
                detected_color = detect_shirt(image_np, shirt_roi   )

                # Draw bounding boxes for shirts
                # if detected_color in ['black', 'white']:
                #     shirt_x, shirt_y, shirt_w, shirt_h = shirt_roi
                #     cv2.rectangle(image_np, (shirt_x, shirt_y), (shirt_x + shirt_w, shirt_y + shirt_h), (0, 0, 255), 2)
                #     cv2.putText(image_np, detected_color, (shirt_x, shirt_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                #                 (0, 0, 255), 1)
                if detected_color == 'black':
                    label = 'child'
                elif detected_color == 'white':
                    age = 23
                    label = f'{gender}, {age}'
                else:
                    label = f'{gender}, {age}'

                # Update gender counters
                if gender == 'Male':
                    male_count += 1
                elif gender == 'Female':
                    female_count += 1
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                rect_x1 = x
                rect_y1 = y - text_height - 10
                rect_x2 = x + text_width - 10
                rect_y2 = y - 5
                cv2.rectangle(image_np, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
                cv2.putText(image_np, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display images and counts
        st.image(image_np, caption="Processed Image", use_column_width=True)
        st.write(f"Number of Males: {male_count}")
        st.write(f"Number of Females: {female_count}")


    # Upload video
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file:
        st.write("Processing video...")

        # Read the video file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            temp_file_path = temp_file.name

        # Open the temporary video file with OpenCV
        video = cv2.VideoCapture(temp_file_path)

        # Create a Streamlit placeholder for video frames
        stframe = st.empty()

        while True:
            ret, frame = video.read()
            if not ret:
                break
            male_count = 0
            female_count = 0
            faces = detect_faces(frame)
            if len(faces)>=2:
                for (x, y, w, h) in faces:
                    age, gender = detect_age_gender(frame[y:y + h, x:x + w])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f'{gender}, {age}'
                    shirt_roi = create_shirt_roi(frame, (x, y, w, h))
                    detected_color = detect_shirt(frame, shirt_roi)

                    # Draw bounding boxes for shirts
                    # if detected_color in ['black', 'white']:
                    #     shirt_x, shirt_y, shirt_w, shirt_h = shirt_roi
                    #     cv2.rectangle(frame, (shirt_x, shirt_y), (shirt_x + shirt_w, shirt_y + shirt_h), (0, 0, 255), 2)
                    #     cv2.putText(frame, detected_color, (shirt_x, shirt_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    #                 (0, 0, 255), 2)

                    if detected_color == 'black':
                        label = 'child'
                    elif detected_color == 'white':
                        age = 23
                        label = f'{gender}, {age}'

                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    rect_x1 = x
                    rect_y1 = y - text_height - 10
                    rect_x2 = x + text_width - 10
                    rect_y2 = y - 5
                    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Update gender counters
                    if gender == 'Male':
                        male_count += 1
                    elif gender == 'Female':
                        female_count += 1

                # Display counts on video frame
                frame_height, frame_width = frame.shape[:2]
                text = f"Males: {male_count} | Females: {female_count}"
                cv2.putText(frame, text, (frame_width - 300, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
                        2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, caption="Processing Video", use_column_width=True)

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()