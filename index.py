import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class AlzheimerDetector:
    def __init__(self, img_height=176, img_width=176, channels=3):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.model = None
        self.class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        self.history = None
        
    def find_dataset_folders(self, base_dir):
        """
        Automatically find dataset folders with flexible structure
        """
        possible_structures = [
            # Structure 1: Direct class folders
            {
                'type': 'direct_classes',
                'classes': self.class_names,
                'path': base_dir
            },
            # Structure 2: Train/Test split
            {
                'type': 'train_test',
                'subfolders': ['train', 'test'],
                'classes': self.class_names,
                'path': base_dir
            },
            # Structure 3: Your specific structure
            {
                'type': 'your_structure',
                'subfolders': ['train', 'test', 'dataset'],
                'classes': self.class_names,
                'path': base_dir
            }
        ]
        
        found_structure = None
        total_images = 0
        
        for structure in possible_structures:
            if structure['type'] == 'direct_classes':
                # Check if class folders exist directly in base_dir
                valid_classes = []
                for class_name in structure['classes']:
                    class_path = os.path.join(structure['path'], class_name)
                    if os.path.exists(class_path):
                        valid_classes.append(class_name)
                
                if len(valid_classes) > 0:
                    found_structure = structure.copy()
                    found_structure['valid_classes'] = valid_classes
                    print(f"✅ Found structure: Direct class folders with classes {valid_classes}")
                    break
                    
            elif structure['type'] in ['train_test', 'your_structure']:
                # Check for subfolder structure
                valid_subfolders = []
                for subfolder in structure['subfolders']:
                    subfolder_path = os.path.join(structure['path'], subfolder)
                    if os.path.exists(subfolder_path):
                        # Check if subfolder has class directories
                        has_classes = any(
                            os.path.exists(os.path.join(subfolder_path, class_name))
                            for class_name in structure['classes']
                        )
                        if has_classes:
                            valid_subfolders.append(subfolder)
                
                if len(valid_subfolders) > 0:
                    found_structure = structure.copy()
                    found_structure['valid_subfolders'] = valid_subfolders
                    print(f"✅ Found structure: {structure['type']} with subfolders {valid_subfolders}")
                    break
        
        return found_structure
    
    def load_images_from_path(self, folder_path, class_name):
        """Load all images from a specific class folder"""
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            return images, labels
            
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.jpeg', '.jpg', '.png', '.jfif')):
                img_path = os.path.join(folder_path, img_file)
                
                try:
                    # Load and preprocess image
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img = img.resize((self.img_width, self.img_height))
                    img_array = np.array(img) / 255.0
                    
                    images.append(img_array)
                    labels.append(class_name)
                    
                except Exception as e:
                    print(f"⚠️ Error loading {img_path}: {e}")
        
        return images, labels
    
    def load_data_flexible(self, base_dir):
        """
        Load data with flexible folder structure detection
        """
        print("🔍 Scanning for dataset structure...")
        structure = self.find_dataset_folders(base_dir)
        
        if structure is None:
            print("❌ No valid dataset structure found!")
            return np.array([]), np.array([])
        
        all_images = []
        all_labels = []
        
        if structure['type'] == 'direct_classes':
            # Load from direct class folders
            for class_name in structure['valid_classes']:
                class_path = os.path.join(base_dir, class_name)
                print(f"📁 Loading from {class_path}...")
                images, labels = self.load_images_from_path(class_path, class_name)
                all_images.extend(images)
                all_labels.extend(labels)
                print(f"   ✅ Loaded {len(images)} images for {class_name}")
                
        elif structure['type'] in ['train_test', 'your_structure']:
            # Load from subfolder structure
            for subfolder in structure['valid_subfolders']:
                subfolder_path = os.path.join(base_dir, subfolder)
                print(f"📂 Scanning {subfolder_path}...")
                
                for class_name in self.class_names:
                    class_path = os.path.join(subfolder_path, class_name)
                    if os.path.exists(class_path):
                        images, labels = self.load_images_from_path(class_path, class_name)
                        all_images.extend(images)
                        all_labels.extend(labels)
                        print(f"   ✅ {class_name}: {len(images)} images")
        
        # Convert to numpy arrays
        if len(all_images) == 0:
            print("❌ No images found in any of the expected locations!")
            return np.array([]), np.array([])
        
        images_array = np.array(all_images)
        
        # Encode labels
        label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        encoded_labels = np.array([label_to_idx[label] for label in all_labels])
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total images: {len(images_array)}")
        print(f"   Image shape: {images_array.shape}")
        
        # Print class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        for class_name, count in zip(unique, counts):
            print(f"   {class_name}: {count} images")
        
        return images_array, encoded_labels
    
    def create_model(self, num_classes):
        """
        Create a CNN model for Alzheimer's detection
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', 
                   input_shape=(self.img_height, self.img_width, self.channels)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train(self, data_dir=None, batch_size=32, epochs=50, learning_rate=0.001):
        """
        Train the Alzheimer detection model
        """
        if data_dir is None:
            # Ask user for dataset path
            data_dir = input("📁 Enter the path to your Alzheimer dataset: ").strip()
            data_dir = data_dir.strip('"')  # Remove quotes if present
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            print(f"❌ Directory '{data_dir}' does not exist!")
            return None, None
        
        # Load data
        print("🔄 Loading dataset...")
        X, y = self.load_data_flexible(data_dir)
        
        if len(X) == 0:
            print("❌ No images found! Please check your dataset path and structure.")
            print("\n💡 Expected structures:")
            print("   Structure 1: Direct class folders")
            print("     dataset/")
            print("     ├── MildDemented/")
            print("     ├── ModerateDemented/")
            print("     ├── NonDemented/")
            print("     └── VeryMildDemented/")
            print("   Structure 2: Train/Test split")
            print("     dataset/")
            print("     ├── train/")
            print("     │   ├── MildDemented/")
            print("     │   └── ...")
            print("     └── test/")
            print("         ├── MildDemented/")
            print("         └── ...")
            return None, None
        
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Validation set: {X_val.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        # Create model
        num_classes = len(self.class_names)
        self.model = self.create_model(num_classes)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n🧠 Model Summary:")
        self.model.summary()
        
        # Callbacks - FIXED: Using .keras extension for newer TensorFlow versions
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.2, patience=10, verbose=1),
            ModelCheckpoint('best_alzheimer_model.keras', save_best_only=True, verbose=1)
        ]
        
        # Train model
        print("\n🚀 Starting training...")
        self.history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        if os.path.exists('best_alzheimer_model.keras'):
            self.model = tf.keras.models.load_model('best_alzheimer_model.keras')
            print("✅ Loaded best model from checkpoint.")
        
        # Evaluate on test set
        print("\n📈 Evaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
        
        return X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model and show detailed metrics
        """
        if self.model is None:
            print("❌ Model not trained yet!")
            return
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\n" + "="*60)
        print("📊 CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Alzheimer Detection', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot training history
        self.plot_training_history()
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("❌ No training history available!")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path):
        """
        Predict Alzheimer's class for a single image
        """
        if self.model is None:
            print("❌ Model not trained yet!")
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
    
    def save_model(self, file_path="alzheimer_detector_model.pkl"):
        """
        Save the trained model as a pickle file
        """
        if self.model is None:
            print("❌ No model to save!")
            return False
        
        try:
            # Create a dictionary with all necessary components
            model_data = {
                'model_architecture': self.model.to_json(),
                'model_weights': self.model.get_weights(),
                'class_names': self.class_names,
                'img_height': self.img_height,
                'img_width': self.img_width,
                'channels': self.channels
            }
            
            # Save to pickle file
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved successfully as {file_path}")
            
            # Also save as .keras for Keras (new format)
            keras_path = file_path.replace('.pkl', '.keras')
            self.model.save(keras_path)
            print(f"✅ Model also saved as Keras format: {keras_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    @classmethod
    def load_model(cls, file_path="alzheimer_detector_model.pkl"):
        """
        Load a saved model from pickle file
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create detector instance
            detector = cls(
                img_height=model_data['img_height'],
                img_width=model_data['img_width'],
                channels=model_data['channels']
            )
            
            detector.class_names = model_data['class_names']
            
            # Rebuild model architecture
            detector.model = tf.keras.models.model_from_json(
                model_data['model_architecture']
            )
            
            # Set weights
            detector.model.set_weights(model_data['model_weights'])
            
            print(f"✅ Model loaded successfully from {file_path}")
            return detector
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Try to load from .keras format as fallback
            keras_path = file_path.replace('.pkl', '.keras')
            if os.path.exists(keras_path):
                print(f"🔄 Trying to load from {keras_path}...")
                try:
                    detector = cls()
                    detector.model = tf.keras.models.load_model(keras_path)
                    detector.class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
                    print(f"✅ Model loaded from {keras_path}")
                    return detector
                except Exception as e2:
                    print(f"❌ Error loading from Keras format: {e2}")
            return None

def main():
    """
    Main function to run the Alzheimer detection training
    """
    print("🧠 ALZHEIMER'S DETECTION MODEL TRAINING")
    print("="*60)
    
    # Initialize detector
    detector = AlzheimerDetector(img_height=176, img_width=176)
    
    # Train the model (user will be prompted for path)
    print("\n📁 Please provide the path to your Alzheimer dataset")
    print("   You can use:")
    print("   - Absolute path: C:/Users/YourName/Documents/Alzheimer_Dataset")
    print("   - Relative path: ./Alzheimer_s Dataset")
    print("   - Or drag and drop the folder into the terminal")
    print()
    
    X_test, y_test = detector.train(
        batch_size=32,
        epochs=50,
        learning_rate=0.001
    )
    
    if X_test is not None:
        # Evaluate the model
        print("\n📊 Evaluating model...")
        detector.evaluate_model(X_test, y_test)
        
        # Save the model
        print("\n💾 Saving model...")
        detector.save_model()
        
        # Test prediction
        print("\n🔍 Testing prediction capability...")
        test_image = input("Enter path to a test image (or press Enter to skip): ").strip()
        
        if test_image and os.path.exists(test_image):
            result = detector.predict_single_image(test_image)
            if result:
                print(f"\n🎯 PREDICTION RESULT:")
                print(f"   Class: {result['class']}")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"\n📊 All probabilities:")
                for class_name, prob in result['all_probabilities'].items():
                    print(f"   {class_name}: {prob:.4f}")
        else:
            print("⏭️  Skipping prediction test")

if __name__ == "__main__":
    main()