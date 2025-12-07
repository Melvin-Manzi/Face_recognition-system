Face Recognition Without Machine Learning — MediaPipe + LBPH Approach

This repository showcases a full facial-recognition workflow built without deep-learning training.
Instead of neural networks, it relies on a blend of:

MediaPipe Face Detection — for efficient, real-time bounding-box detection

LBPH (Local Binary Patterns Histograms) — for classic, feature-based face recognition

The project demonstrates how traditional AI techniques can perform detection and identification without modern ML models.

🚀 What This Project Includes

⚡ Live face detection powered by MediaPipe

🧩 LBPH-based face recognition

🗂️ Add and train multiple users

🎥 Real-time identity prediction from webcam

💻 Works on CPU — no GPU or heavy frameworks needed

📁 Directory Layout
project/
│
├── dataset/
│ ├── Person1/
│ ├── Person2/
│ └── ...
│
├── models/
│ ├── lbph_face_model.xml
│ └── label_map.json
│
├── capture.py # Gather training images
├── train.py # Build LBPH recognition model
├── predict.py # Run live recognition
└── README.md

🛠️ 1. Setup & Installation

Install the required Python packages:

pip install opencv-python
pip install opencv-contrib-python
pip install mediapipe

Important:
opencv-contrib-python is necessary because it contains the LBPH face recognizer module.

🎥 2. Collect Training Images

Start the image-collection script:

python capture.py

Then:

Type the name of the person you want to register (e.g., Joyeuse).

Face the webcam.

Images are saved automatically inside:
dataset/<person_name>/

Press Q when finished.

Repeat the process for at least two people.

🧠 3. Train the LBPH Recognition Model

Run the training script:

python train.py

This step will:

Load images from dataset/

Assign label IDs for each user

Train the LBPH recognizer

Output:

models/lbph_face_model.xml

models/label_map.json

Upon completion, you should see:

Training completed! Model saved.

👁️ 4. Perform Live Face Recognition

Start real-time recognition with:

python predict.py

What happens:

MediaPipe detects facial bounding boxes

Each detected face is cropped

LBPH predicts the identity

Name + confidence score are shown on the video stream

Press Q to exit

🎬 5. Required Video Submission

Your final submission must contain a 20–40 second demonstration video showing:

Two different individuals

MediaPipe successfully detecting both faces

LBPH correctly recognizing each person

🔍 6. How the System Works
Face Detection (MediaPipe)

MediaPipe provides a fast, lightweight detector capable of returning:

Bounding box coordinates

Facial landmarks

Detection confidence

The bounding box is used to extract the face region for recognition.

Face Recognition (LBPH)

LBPH works by:

Converting images to grayscale

Extracting texture-based patterns

Generating histogram descriptors

Comparing them to stored representations

It produces:

A predicted user label

A confidence value

LBPH is ideal for small datasets and performs well in real-time.

🎯 7. Improving Recognition Accuracy

For best results:

Gather 40–100 images per person

Provide good lighting conditions

Use a simple background

Capture faces from left, right, and center

Avoid motion blur

👤 Author

Name: Melvin
Project: AI Without ML — Week 13
Institution: Rwanda Coding Academy
