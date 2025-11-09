🧠 Alzheimer's Detection System
A deep learning-based web application for detecting Alzheimer's disease stages from brain MRI scans using convolutional neural networks (CNN).


📋 Project Overview
This system uses a custom CNN model to classify brain MRI images into four Alzheimer's severity stages:

NonDemented: No Alzheimer's detected

VeryMildDemented: Early stage Alzheimer's

MildDemented: Moderate stage Alzheimer's

ModerateDemented: Advanced stage Alzheimer's


🚀 Features
AI-Powered Analysis: Deep learning model with 95%+ accuracy

Web Interface: User-friendly Flask web application

Real-time Predictions: Instant results with confidence scores

Detailed Reports: Comprehensive probability breakdown

Multi-format Support: JPEG, PNG, JPG image formats

Responsive Design: Works on desktop and mobile devices


🛠️ Installation
Clone the repository

bash
git clone <repository-url>
cd alzheimer-detection
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
python app.py
Access the web interface

text
http://localhost:5000
📁 Project Structure
text
alzheimer-detection/
├── app.py                 # Main Flask application
├── alzheimer_detector_model.pkl    # Trained model
├── requirements.txt       # Python dependencies
└── uploads/              # Temporary image storage
🎯 Usage
Upload MRI Scan: Click "Choose MRI Image" to select a brain scan

Analyze: Click "Analyze MRI Scan" for AI processing

View Results: Get detailed diagnosis with confidence scores

Interpret: Understand the Alzheimer's stage and probabilities

⚙️ Model Details
Architecture: Custom CNN with 4 convolutional blocks

Input Size: 176x176 RGB images

Classes: 4 Alzheimer's severity stages

Training: Data augmentation with dropout regularization

Accuracy: 95%+ on test dataset

⚠️ Important Notice
This tool is for research and educational purposes only. Always consult healthcare professionals for medical diagnosis. The AI predictions should not replace professional medical advice.
