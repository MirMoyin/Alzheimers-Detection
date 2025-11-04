# predict_alzheimer.py
import os
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf

class AlzheimerPredictor:
    def __init__(self, model_path="alzheimer_detector_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.img_height = None
        self.img_width = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from pickle file or .keras format"""
        try:
            # First try to load from pickle
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.class_names = model_data['class_names']
                self.img_height = model_data['img_height']
                self.img_width = model_data['img_width']
                
                # Rebuild model
                self.model = tf.keras.models.model_from_json(model_data['model_architecture'])
                self.model.set_weights(model_data['model_weights'])
                
                print("✅ Model loaded successfully from pickle!")
                
            # If pickle doesn't exist, try .keras format
            elif os.path.exists('best_alzheimer_model.keras'):
                self.model = tf.keras.models.load_model('best_alzheimer_model.keras')
                self.class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
                self.img_height = 176
                self.img_width = 176
                print("✅ Model loaded successfully from .keras format!")
                
            else:
                # Try the .keras version of the pickle path
                keras_path = self.model_path.replace('.pkl', '.keras')
                if os.path.exists(keras_path):
                    self.model = tf.keras.models.load_model(keras_path)
                    self.class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
                    self.img_height = 176
                    self.img_width = 176
                    print("✅ Model loaded successfully from .keras format!")
                else:
                    print("❌ No model file found!")
                    return
            
            print(f"📋 Classes: {self.class_names}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def predict(self, image_path):
        """Predict Alzheimer's class for a new image"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        try:
            # Load and preprocess image
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize((self.img_width, self.img_height))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            result = {
                'class': self.class_names[predicted_class],
                'confidence': float(confidence),
                'all_probabilities': {
                    self.class_names[i]: float(prediction[0][i]) 
                    for i in range(len(self.class_names))
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error predicting image: {e}")
            return None

def main():
    # Initialize predictor
    predictor = AlzheimerPredictor()
    
    while True:
        print("\n🧠 Alzheimer's Detection Predictor")
        print("="*40)
        image_path = input("Enter the path to the brain MRI image (or 'quit' to exit): ").strip()
        
        if image_path.lower() == 'quit':
            break
            
        if os.path.exists(image_path):
            result = predictor.predict(image_path)
            
            if result:
                print(f"\n🎯 PREDICTION RESULTS:")
                print(f"   Class: {result['class']}")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"\n📊 Detailed Probabilities:")
                for class_name, prob in result['all_probabilities'].items():
                    print(f"   {class_name}: {prob:.4f}")
        else:
            print("❌ Image file not found! Please check the path.")

if __name__ == "__main__":
    main()