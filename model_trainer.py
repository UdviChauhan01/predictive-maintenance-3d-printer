import numpy as np
import pandas as pd
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
except ImportError as e:
    print(f"Warning: Some ML libraries not available: {e}")

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")

class PrinterPredictiveModel:
    """
    Predictive maintenance model for 3D printers
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = f"models/{model_type}_model.pkl"
        self.scaler_path = f"models/{model_type}_scaler.pkl"
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_model(self, input_shape: Tuple[int, ...] = None):
        """Create the specified model type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'lightgbm':
            self.model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42, probability=True)
        elif self.model_type == 'neural_network':
            if input_shape is None:
                raise ValueError("Input shape required for neural network")
            self.model = self._create_neural_network(input_shape)
        elif self.model_type == 'lstm':
            if input_shape is None:
                raise ValueError("Input shape required for LSTM")
            self.model = self._create_lstm_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")    
    def _create_neural_network(self, input_shape: Tuple[int, ...]):
        """Create a neural network model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ]) 
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_lstm_model(self, input_shape: Tuple[int, ...]):
        """Create an LSTM model for time series prediction"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: List[str] = None):
        """Train the model"""
        if self.model is None:
            if self.model_type in ['neural_network', 'lstm']:
                self.create_model(input_shape=X_train.shape[1:])
            else:
                self.create_model()
        
        self.feature_names = feature_names or []
        
        if self.model_type in ['neural_network', 'lstm']:
            # For deep learning models
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(self.model_path.replace('.pkl', '.h5'), 
                              save_best_only=True, monitor='val_loss')
            ]
            
            validation_data = (X_val, y_val) if X_val is not None else None
            
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
        else:
            # For traditional ML models
            if X_val is not None:
                # Use validation set for early stopping
                self.model.fit(X_train, y_train)
                val_score = self.model.score(X_val, y_val)
                self.logger.info(f"Validation accuracy: {val_score:.4f}")
            else:
                self.model.fit(X_train, y_train)
            
            # Save the model
            self.save_model()
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.model_type in ['neural_network', 'lstm']:
            return (self.model.predict(X) > 0.5).astype(int)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.model_type in ['neural_network', 'lstm']:
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        # Get feature importance for tree-based models
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if self.model_type in ['neural_network', 'lstm']:
            self.model.save(self.model_path.replace('.pkl', '.h5'))
        else:
            joblib.dump(self.model, self.model_path)
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'model_path': self.model_path,
            'scaler_path': self.scaler_path
        }
        
        with open(self.model_path.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model"""
        if self.model_type in ['neural_network', 'lstm']:
            self.model = keras.models.load_model(self.model_path.replace('.pkl', '.h5'))
        else:
            self.model = joblib.load(self.model_path)
        
        self.scaler = joblib.load(self.scaler_path)
        
        # Load metadata
        metadata_path = self.model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
        
        self.logger.info(f"Model loaded from {self.model_path}")

class ModelEnsemble:
    """
    Ensemble of multiple models for improved prediction
    """
    
    def __init__(self, model_types: List[str] = None):
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'xgboost']
            
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.weights = {}
        
        for model_type in model_types:
            self.models[model_type] = PrinterPredictiveModel(model_type)
            self.weights[model_type] = 1.0 / len(model_types)
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None,
                  feature_names: List[str] = None):
        """Train all models in the ensemble"""
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            model.train(X_train, y_train, X_val, y_val, feature_names)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict_proba(X)
            if len(pred.shape) > 1:
                pred = pred[:, 1]  # Get positive class probability
            predictions.append(pred * self.weights[name])
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities"""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict_proba(X)
            if len(pred.shape) > 1:
                pred = pred[:, 1]  # Get positive class probability
            predictions.append(pred * self.weights[name])
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

def train_models_from_data(data_file: str, output_dir: str = 'models'):
    """Complete training pipeline"""
    from data_preprocessor import PrinterDataPreprocessor
    
    # Preprocess data
    preprocessor = PrinterDataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(data_file)
    
    if not processed_data:
        print("No data to train on")
        return
    
    # Split data
    X = processed_data['regular_features']
    y = processed_data['regular_targets']

    # 🚨 Fix NaN values using mean imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
    )
    # ✅ Step 1: Print class distribution
    import numpy as np
    print("y_train classes:", np.unique(y_train, return_counts=True))
    print("y_test classes:", np.unique(y_test, return_counts=True))

    # Train different model types
    model_types = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        model = PrinterPredictiveModel(model_type)
        model.train(X_train, y_train, X_test, y_test, processed_data['feature_names'])
        
        # Evaluate
        eval_results = model.evaluate(X_test, y_test)
        results[model_type] = eval_results
        
        print(f"{model_type} Results:")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall: {eval_results['recall']:.4f}")
        print(f"  F1 Score: {eval_results['f1_score']:.4f}")
    
    # Train ensemble
    print("\nTraining ensemble...")
    ensemble = ModelEnsemble(model_types)
    ensemble.train_all(X_train, y_train, X_test, y_test, processed_data['feature_names'])
    
    # Evaluate ensemble
    ensemble_pred = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("3D Printer Predictive Maintenance Model Trainer")
    print("=" * 50)
    
    # Check if sample data exists
    if os.path.exists('sample_printer_data.json'):
        print("Training models on sample data...")
        results = train_models_from_data('sample_printer_data.json')
        
        if results:
            print("\nTraining completed successfully!")
            print("Models saved in 'models/' directory")
    else:
        print("No sample data found. Run data_preprocessor.py first to generate sample data.") 