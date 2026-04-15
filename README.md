 Driver Drowsiness Detection Using ML	

📌 Project Overview

Driver Drowsiness Detection is a real-time computer vision system designed to detect driver fatigue and prevent road accidents. The system monitors the driver's face and eye state using a webcam and classifies the driver's condition as Alert (Open Eyes) or Drowsy (Closed Eyes).

This project uses both Convolutional Neural Network (CNN) and Support Vector Machine (SVM) models for training and testing. The CNN model was deployed for real-time detection due to its higher accuracy.



🎯 Objectives

- Detect driver drowsiness in real time
- Improve road safety by alerting the driver
- Compare performance of CNN and SVM models
- Deploy the best-performing model (CNN)
- Generate evaluation outputs such as accuracy graph and confusion matrix



🧠 Technologies Used

- Programming Language: Python

- Libraries:
  
  - OpenCV
  - NumPy
  - TensorFlow / Keras
  - Scikit-learn
  - Matplotlib
  - Seaborn

- Models Used:
  
  - Convolutional Neural Network (CNN)
  - Support Vector Machine (SVM)

- Development Tool: VS Code

- Hardware: Webcam



📁 Project Structure

Driver-Drowsiness-Detection/
│
├── dataset/
│   ├── train/
│   │   ├── alert/
│   │   └── drowsy/
│   │
│   └── test/
│       ├── alert/
│       └── drowsy/
│
├── Train_CNN.py
├── train_svm.py
├── main.py
│
├── confusion_matrix_cnn.py
├── accuracy_graph.py
│
├── model.h5
├── svm_model.pkl
│
├── accuracy_graph.png
├── confusion_matrix_cnn.png
│
├── requirements.txt
├── README.md
└── .gitignore



⚙️ Working Methodology

1. Download the eye image dataset (Open and Closed eyes).
2. Split dataset into training and testing folders.
3. Preprocess images (resize and normalize).
4. Train CNN model using training dataset.
5. Train SVM model for comparison.
6. Evaluate models using accuracy graph and confusion matrix.
7. Deploy CNN model for real-time drowsiness detection.


🧪 Model Details

🔹 CNN Model ("model.h5")

- Used for image classification
- Deep learning-based approach
- Higher accuracy than SVM
- Used for real-time detection

🔹 SVM Model ("svm_model.pkl")

- Machine learning model
- Used for performance comparison
- Trained using grayscale image features



📊 Results

CNN Accuracy: 95%
SVM Accuracy: 88%
 

Outputs Generated:

- Accuracy Graph
- Confusion Matrix
- Real-Time Detection Output



🚨 Features

- Real-time driver monitoring
- Eye state classification
- CNN vs SVM performance comparison
- Accuracy graph visualization
- Confusion matrix generation
- High-accuracy detection system


▶️ How to Run the Project

Step 1 — Install Required Libraries

pip install -r requirements.txt



Step 2 — Train CNN Model

python Train_CNN.py

Output:

- "model.h5"
- "accuracy_graph.png"


Step 3 — Train SVM Model

python train_svm.py

Output:

- "svm_model.pkl"



Step 4 — Generate Confusion Matrix

python confusion_matrix_cnn.py

Output:

- "confusion_matrix_cnn.png"



Step 5 — Run Real-Time Detection

python main.py



📁 Dataset

This project uses the MRL Eye Dataset, a widely used dataset for eye state classification.

Dataset Link:
https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset

Dataset contains:

- Open Eye Images (Alert)
- Closed Eye Images (Drowsy)
- More than 4000 eye images

After downloading:

Split dataset into:

dataset/train/alert
dataset/train/drowsy
dataset/test/alert
dataset/test/drowsy



📷 Outputs

This project generates:

- Accuracy Graph ("accuracy_graph.png")
- Confusion Matrix ("confusion_matrix_cnn.png")
- Real-Time Detection Window





🔍 Applications

- Driver safety systems
- Smart vehicles
- Road accident prevention
- Automotive safety monitoring
- Intelligent transportation systems



🚀 Future Enhancements

- Add yawning detection
- Add buzzer alarm system
- Deploy on Raspberry Pi
- Improve accuracy using larger datasets
- Add mobile app integration



👨‍💻 Author

Harsha Gopal
B.E Robotics and AI
Jawaharlal Nehru New College of Engineering (JNNCE)

📧 Email:
harshagopal951@gmail.com

🔗 LinkedIn:
https://www.linkedin.com/in/harsha-harshg-784b39249



⭐ Acknowledgment

This project was developed as part of academic learning in Machine Learning, Computer Vision, and Embedded Vision systems. 
