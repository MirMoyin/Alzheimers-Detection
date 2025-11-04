from flask import Flask, render_template_string, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from datetime import datetime
import base64
import io

app = Flask(__name__)
app.secret_key = 'alzheimer_detection_secret_key'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class AlzheimerPredictor:
    def __init__(self, model_path='C:\MY FOLDER\project\Project 4\\alzheimer_detector_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.img_height = None
        self.img_width = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from pickle file"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.class_names = model_data['class_names']
            self.img_height = model_data['img_height']
            self.img_width = model_data['img_width']
            
            # Rebuild model
            self.model = tf.keras.models.model_from_json(model_data['model_architecture'])
            self.model.set_weights(model_data['model_weights'])
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Try to load from .keras format as fallback
            try:
                keras_path = self.model_path.replace('.pkl', '.keras')
                self.model = tf.keras.models.load_model(keras_path)
                self.class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
                self.img_height = 176
                self.img_width = 176
                print("✅ Model loaded from .keras format!")
            except Exception as e2:
                print(f"❌ Error loading from Keras format: {e2}")
    
    def preprocess_image(self, image_path):
        """Preprocess the image for prediction"""
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize((self.img_width, self.img_height))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """Predict Alzheimer's class for an image"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return {"error": "Error processing image"}
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            # Get all probabilities
            all_probabilities = {
                self.class_names[i]: float(prediction[0][i]) 
                for i in range(len(self.class_names))
            }
            
            # Determine result type
            predicted_class = self.class_names[predicted_class_idx]
            has_alzheimer = predicted_class != 'NonDemented'
            severity = self.get_severity_level(predicted_class)
            
            result = {
                'class': predicted_class,
                'confidence': float(confidence),
                'has_alzheimer': has_alzheimer,
                'severity': severity,
                'all_probabilities': all_probabilities,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            print(f"Error predicting image: {e}")
            return {"error": str(e)}
    
    def get_severity_level(self, class_name):
        """Convert class name to severity level"""
        severity_map = {
            'NonDemented': 'No Alzheimer\'s Detected',
            'VeryMildDemented': 'Very Mild Alzheimer\'s',
            'MildDemented': 'Mild Alzheimer\'s',
            'ModerateDemented': 'Moderate Alzheimer\'s'
        }
        return severity_map.get(class_name, class_name)

# Initialize predictor
predictor = AlzheimerPredictor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML Templates as strings
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alzheimer's Detection - MRI Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            line-height: 1.6; color: #333; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; color: white; }
        .logo { display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px; }
        .logo i { font-size: 2.5rem; color: #fff; }
        .logo h1 { font-size: 2.5rem; font-weight: 700; }
        .subtitle { font-size: 1.2rem; opacity: 0.9; }
        .upload-card { 
            background: white; padding: 40px; border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1); text-align: center; 
            max-width: 500px; width: 100%; margin: 0 auto;
        }
        .upload-icon { font-size: 4rem; color: #667eea; margin-bottom: 20px; }
        .upload-card h2 { color: #333; margin-bottom: 15px; font-size: 1.8rem; }
        .upload-info { color: #666; margin-bottom: 30px; font-size: 1.1rem; }
        .file-input-container { margin-bottom: 25px; }
        .file-input-container input[type="file"] { display: none; }
        .file-input-label { 
            display: inline-flex; align-items: center; gap: 10px; 
            padding: 15px 30px; background: #667eea; color: white; 
            border-radius: 10px; cursor: pointer; transition: all 0.3s ease; 
            font-weight: 600; 
        }
        .file-input-label:hover { background: #5a6fd8; transform: translateY(-2px); }
        .file-name { margin-top: 10px; color: #666; font-style: italic; }
        .analyze-btn { 
            width: 100%; padding: 15px; background: #27ae60; color: white; 
            border: none; border-radius: 10px; font-size: 1.1rem; 
            font-weight: 600; cursor: pointer; transition: all 0.3s ease; 
            display: flex; align-items: center; justify-content: center; gap: 10px; 
        }
        .analyze-btn:hover { background: #219653; transform: translateY(-2px); }
        .analyze-btn:disabled { background: #bdc3c7; cursor: not-allowed; transform: none; }
        .loading { text-align: center; padding: 20px; display: none; }
        .spinner { 
            border: 4px solid #f3f3f3; border-top: 4px solid #667eea; 
            border-radius: 50%; width: 40px; height: 40px; 
            animation: spin 1s linear infinite; margin: 0 auto 15px; 
        }
        @keyframes spin { 
            0% { transform: rotate(0deg); } 
            100% { transform: rotate(360deg); } 
        }
        .info-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; margin-top: 40px; 
        }
        .info-card { 
            background: white; padding: 30px; border-radius: 15px; 
            text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.1); 
        }
        .info-card i { font-size: 2.5rem; color: #667eea; margin-bottom: 15px; }
        .footer { text-align: center; margin-top: 50px; color: white; opacity: 0.8; }
        .disclaimer { font-size: 0.9rem; margin-top: 10px; opacity: 0.7; }
        .alert { 
            padding: 15px; border-radius: 10px; margin-bottom: 20px; 
            display: flex; align-items: center; gap: 10px; 
        }
        .alert-error { 
            background: #f8d7da; color: #721c24; 
            border-left: 5px solid #dc3545; 
        }
        @media (max-width: 768px) {
            .container { padding: 15px; }
            .logo h1 { font-size: 2rem; }
            .upload-card { padding: 25px; }
        }
    </style>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="logo">
                <i class="fas fa-brain"></i>
                <h1>Alzheimer's Detection</h1>
            </div>
            <p class="subtitle">AI-Powered MRI Analysis for Early Detection</p>
        </header>

        <main class="main-content">
            <div class="upload-section">
                <div class="upload-card">
                    <div class="upload-icon">
                        <i class="fas fa-cloud-upload-alt"></i>
                    </div>
                    <h2>Upload MRI Scan</h2>
                    <p class="upload-info">
                        Upload a brain MRI image (JPG, PNG, JPEG) to analyze for Alzheimer's disease
                    </p>

                    {% with messages = get_flashed_messages(with_categories=true) %}
                        {% if messages %}
                            {% for category, message in messages %}
                                <div class="alert alert-{{ category }}">
                                    <i class="fas fa-exclamation-circle"></i>
                                    {{ message }}
                                </div>
                            {% endfor %}
                        {% endif %}
                    {% endwith %}

                    <form id="uploadForm" action="/predict" method="post" enctype="multipart/form-data">
                        <div class="file-input-container">
                            <input type="file" id="fileInput" name="file" accept=".jpg,.jpeg,.png" required>
                            <label for="fileInput" class="file-input-label">
                                <i class="fas fa-file-image"></i>
                                <span>Choose MRI Image</span>
                            </label>
                            <div id="fileName" class="file-name"></div>
                        </div>
                        
                        <button type="submit" class="analyze-btn" id="analyzeBtn">
                            <i class="fas fa-search"></i>
                            Analyze MRI Scan
                        </button>
                    </form>

                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p>Analyzing MRI scan... This may take a few seconds.</p>
                    </div>
                </div>
            </div>

            <div class="info-section">
                <div class="info-grid">
                    <div class="info-card">
                        <i class="fas fa-shield-alt"></i>
                        <h3>Accurate Detection</h3>
                        <p>Powered by deep learning algorithms trained on thousands of MRI scans</p>
                    </div>
                    <div class="info-card">
                        <i class="fas fa-bolt"></i>
                        <h3>Fast Analysis</h3>
                        <p>Get results in seconds with our optimized AI model</p>
                    </div>
                    <div class="info-card">
                        <i class="fas fa-chart-bar"></i>
                        <h3>Detailed Report</h3>
                        <p>Receive comprehensive analysis with confidence scores</p>
                    </div>
                </div>
            </div>
        </main>

        <footer class="footer">
            <p>&copy; 2024 Alzheimer's Detection System. For research and educational purposes.</p>
            <p class="disclaimer">
                <strong>Disclaimer:</strong> This tool is for research purposes only. 
                Always consult healthcare professionals for medical diagnosis.
            </p>
        </footer>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const fileInput = document.getElementById('fileInput');
            const fileName = document.getElementById('fileName');
            const uploadForm = document.getElementById('uploadForm');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');

            if (fileInput) {
                fileInput.addEventListener('change', function(e) {
                    if (this.files && this.files[0]) {
                        fileName.textContent = this.files[0].name;
                        
                        const fileSize = this.files[0].size;
                        const maxSize = 16 * 1024 * 1024;
                        
                        if (fileSize > maxSize) {
                            alert('File size exceeds 16MB limit. Please choose a smaller file.');
                            this.value = '';
                            fileName.textContent = '';
                            return;
                        }
                    }
                });
            }

            if (uploadForm) {
                uploadForm.addEventListener('submit', function(e) {
                    if (!fileInput.files || !fileInput.files[0]) {
                        e.preventDefault();
                        alert('Please select an image file first.');
                        return;
                    }

                    analyzeBtn.disabled = true;
                    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
                    loading.style.display = 'block';
                });
            }
        });
    </script>
</body>
</html>
'''

RESULT_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Result - Alzheimer's Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            line-height: 1.6; color: #333; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { 
            display: flex; justify-content: space-between; align-items: center; 
            margin-bottom: 40px; color: white; 
        }
        .logo { display: flex; align-items: center; gap: 15px; }
        .logo i { font-size: 2.5rem; color: #fff; }
        .logo h1 { font-size: 2.5rem; font-weight: 700; }
        .back-btn { 
            display: inline-flex; align-items: center; gap: 8px; color: white; 
            text-decoration: none; padding: 10px 20px; 
            background: rgba(255,255,255,0.2); border-radius: 25px; 
            transition: all 0.3s ease; 
        }
        .back-btn:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
        .result-section { 
            display: grid; grid-template-columns: 2fr 1fr; gap: 30px; 
        }
        .result-card { 
            background: white; border-radius: 20px; padding: 30px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
        }
        .result-header { 
            text-align: center; margin-bottom: 30px; padding-bottom: 20px; 
            border-bottom: 2px solid #f8f9fa; 
        }
        .result-header h2 { color: #333; margin-bottom: 10px; }
        .timestamp { color: #666; font-style: italic; }
        .diagnosis-card { 
            display: flex; align-items: center; gap: 20px; padding: 25px; 
            border-radius: 15px; margin-bottom: 30px; 
        }
        .diagnosis-card.positive { 
            background: #ffeaea; border-left: 5px solid #e74c3c; 
        }
        .diagnosis-card.negative { 
            background: #e8f6ef; border-left: 5px solid #27ae60; 
        }
        .diagnosis-icon { font-size: 3rem; }
        .diagnosis-card.positive .diagnosis-icon { color: #e74c3c; }
        .diagnosis-card.negative .diagnosis-icon { color: #27ae60; }
        .diagnosis-content h3 { margin-bottom: 5px; font-size: 1.5rem; }
        .confidence { color: #666; font-size: 1.1rem; }
        .probability-chart { margin-bottom: 30px; }
        .probability-chart h4 { margin-bottom: 20px; color: #333; }
        .probability-item { margin-bottom: 15px; }
        .probability-label { 
            display: flex; justify-content: space-between; margin-bottom: 5px; 
            font-weight: 600; 
        }
        .class-name { text-transform: capitalize; }
        .probability-bar { 
            background: #f8f9fa; border-radius: 10px; height: 10px; 
            overflow: hidden; 
        }
        .probability-fill { 
            height: 100%; border-radius: 10px; transition: width 0.5s ease; 
        }
        .interpretation { margin-bottom: 30px; }
        .interpretation h4 { margin-bottom: 15px; color: #333; }
        .alert { 
            padding: 15px; border-radius: 10px; 
            display: flex; align-items: flex-start; gap: 10px; 
        }
        .alert-warning { 
            background: #fff3cd; border-left: 5px solid #ffc107; color: #856404; 
        }
        .alert-success { 
            background: #d1edff; border-left: 5px solid #3498db; color: #155724; 
        }
        .action-buttons { display: flex; gap: 15px; flex-wrap: wrap; }
        .btn { 
            display: inline-flex; align-items: center; gap: 8px; 
            padding: 12px 25px; border: none; border-radius: 10px; 
            text-decoration: none; font-weight: 600; cursor: pointer; 
            transition: all 0.3s ease; 
        }
        .btn-primary { background: #667eea; color: white; }
        .btn-primary:hover { background: #5a6fd8; transform: translateY(-2px); }
        .btn-secondary { background: #95a5a6; color: white; }
        .btn-secondary:hover { background: #7f8c8d; transform: translateY(-2px); }
        .info-panel { 
            background: white; border-radius: 15px; padding: 25px; 
            box-shadow: 0 10px 20px rgba(0,0,0,0.1); align-self: start; 
        }
        .info-panel h4 { 
            margin-bottom: 20px; color: #333; 
            border-bottom: 2px solid #f8f9fa; padding-bottom: 10px; 
        }
        .info-item { padding: 10px 0; border-bottom: 1px solid #f8f9fa; }
        .info-item:last-child { border-bottom: none; }
        .disclaimer-box { 
            margin-top: 25px; padding: 15px; background: #fff3cd; 
            border-radius: 10px; border-left: 5px solid #ffc107; 
        }
        .disclaimer-box h5 { 
            color: #856404; margin-bottom: 10px; 
            display: flex; align-items: center; gap: 8px; 
        }
        .disclaimer-box p { color: #856404; font-size: 0.9rem; }
        .footer { text-align: center; margin-top: 50px; color: white; opacity: 0.8; }
        @media (max-width: 768px) {
            .result-section { grid-template-columns: 1fr; }
            .action-buttons { flex-direction: column; }
            .diagnosis-card { flex-direction: column; text-align: center; }
            .header { flex-direction: column; gap: 20px; text-align: center; }
        }
    </style>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="logo">
                <i class="fas fa-brain"></i>
                <h1>Alzheimer's Detection</h1>
            </div>
            <a href="/" class="back-btn">
                <i class="fas fa-arrow-left"></i>
                New Analysis
            </a>
        </header>

        <main class="main-content">
            <div class="result-section">
                <div class="result-card">
                    <div class="result-header">
                        <h2>MRI Analysis Result</h2>
                        <p class="timestamp">Analyzed on: {{ result.timestamp }}</p>
                    </div>

                    <div class="result-main">
                        <div class="diagnosis-card {{ 'positive' if result.has_alzheimer else 'negative' }}">
                            <div class="diagnosis-icon">
                                {% if result.has_alzheimer %}
                                    <i class="fas fa-exclamation-triangle"></i>
                                {% else %}
                                    <i class="fas fa-check-circle"></i>
                                {% endif %}
                            </div>
                            <div class="diagnosis-content">
                                <h3>Diagnosis: {{ result.severity }}</h3>
                                <p class="confidence">
                                    Confidence: <strong>{{ "%.2f"|format(result.confidence * 100) }}%</strong>
                                </p>
                            </div>
                        </div>

                        <div class="probability-chart">
                            <h4>Detailed Probabilities</h4>
                            <div class="probability-bars">
                                {% for class_name, probability in result.all_probabilities.items() %}
                                <div class="probability-item">
                                    <div class="probability-label">
                                        <span class="class-name">{{ class_name }}</span>
                                        <span class="probability-value">{{ "%.2f"|format(probability * 100) }}%</span>
                                    </div>
                                    <div class="probability-bar">
                                        <div class="probability-fill" 
                                             style="width: {{ probability * 100 }}%; 
                                                    background: {% if class_name == result.class %} 
                                                                    {% if result.has_alzheimer %}#e74c3c{% else %}#27ae60{% endif %}
                                                                 {% else %}#3498db{% endif %};">
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>

                        <div class="interpretation">
                            <h4>Interpretation</h4>
                            <div class="interpretation-content">
                                {% if result.has_alzheimer %}
                                    <div class="alert alert-warning">
                                        <i class="fas fa-exclamation-triangle"></i>
                                        <strong>Potential Alzheimer's detected.</strong> 
                                        The MRI analysis suggests {{ result.severity|lower }}. 
                                        Please consult with a healthcare professional for comprehensive evaluation.
                                    </div>
                                {% else %}
                                    <div class="alert alert-success">
                                        <i class="fas fa-check-circle"></i>
                                        <strong>No significant Alzheimer's indicators detected.</strong> 
                                        The MRI analysis appears normal. However, regular check-ups are recommended.
                                    </div>
                                {% endif %}
                            </div>
                        </div>

                        <div class="action-buttons">
                            <a href="/" class="btn btn-primary">
                                <i class="fas fa-redo"></i>
                                Analyze Another Image
                            </a>
                            <button onclick="window.print()" class="btn btn-secondary">
                                <i class="fas fa-print"></i>
                                Print Report
                            </button>
                        </div>
                    </div>
                </div>

                <div class="info-panel">
                    <h4>Understanding the Results</h4>
                    <div class="info-content">
                        <div class="info-item">
                            <strong>NonDemented:</strong> No significant signs of Alzheimer's disease
                        </div>
                        <div class="info-item">
                            <strong>VeryMildDemented:</strong> Early, minimal cognitive impairment
                        </div>
                        <div class="info-item">
                            <strong>MildDemented:</strong> Moderate cognitive decline
                        </div>
                        <div class="info-item">
                            <strong>ModerateDemented:</strong> Significant cognitive impairment
                        </div>
                    </div>
                    
                    <div class="disclaimer-box">
                        <h5><i class="fas fa-info-circle"></i> Important Notice</h5>
                        <p>
                            This analysis is generated by an AI model and is intended for research purposes only. 
                            It should not be used as a substitute for professional medical diagnosis. 
                            Always consult qualified healthcare providers for medical advice.
                        </p>
                    </div>
                </div>
            </div>
        </main>

        <footer class="footer">
            <p>&copy; 2024 Alzheimer's Detection System. For research and educational purposes.</p>
        </footer>
    </div>

    <script>
        // Animate probability bars
        document.addEventListener('DOMContentLoaded', function() {
            const probabilityBars = document.querySelectorAll('.probability-fill');
            probabilityBars.forEach(bar => {
                const originalWidth = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = originalWidth;
                }, 100);
            });
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect('/')
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = predictor.predict(filepath)
            
            if 'error' in result:
                return f"Error: {result['error']}"
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return render_template_string(RESULT_HTML, result=result)
            
        except Exception as e:
            return f"Error processing file: {str(e)}"
    
    else:
        return "Invalid file type. Please upload PNG, JPG, or JPEG images."

@app.errorhandler(413)
def too_large(e):
    return "File too large. Please upload images smaller than 16MB."

if __name__ == '__main__':
    print("🚀 Starting Alzheimer's Detection Web Application...")
    print("📧 Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)