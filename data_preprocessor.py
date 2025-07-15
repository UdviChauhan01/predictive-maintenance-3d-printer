import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
from datetime import datetime, timedelta
import random


class PrinterDataPreprocessor:
    """
    Data preprocessing for 3D printer sensor data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = []
        self.target_columns = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_json_data(self, filename: str) -> pd.DataFrame:
        """Load JSON data and convert to DataFrame"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Flatten nested JSON structure
            flattened_data = []
            for record in data:
                flat_record = {
                    'timestamp': record['timestamp'],
                    'hotend_temp': record['temperature']['hotend'],
                    'bed_temp': record['temperature']['bed'],
                    'ambient_temp': record['temperature']['ambient'],
                    'vibration_x': record['vibration']['x_axis'],
                    'vibration_y': record['vibration']['y_axis'],
                    'vibration_z': record['vibration']['z_axis'],
                    'motor_current_x': record['motor_current']['x_motor'],
                    'motor_current_y': record['motor_current']['y_motor'],
                    'motor_current_z': record['motor_current']['z_motor'],
                    'motor_current_extruder': record['motor_current']['extruder'],
                    'position_x': record['position']['x'],
                    'position_y': record['position']['y'],
                    'position_z': record['position']['z'],
                    'is_printing': record['print_status']['is_printing'],
                    'layer_height': record['print_status']['layer_height'],
                    'print_progress': record['print_status']['print_progress'],
                    'filament_used': record['print_status']['filament_used'],
                    'belt_tension': record['maintenance_indicators']['belt_tension'],
                    'nozzle_wear': record['maintenance_indicators']['nozzle_wear'],
                    'bed_level': record['maintenance_indicators']['bed_level'],
                    'extruder_clogging': record['maintenance_indicators']['extruder_clogging']
                }
                flattened_data.append(flat_record)
            
            df = pd.DataFrame(flattened_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by removing outliers and handling missing values"""
        if df.empty:
            return df
        
        # Remove rows with all null values
        df = df.dropna(how='all')
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with bounds
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Ensure boolean columns are properly typed
        if 'is_printing' in df.columns:
            df['is_printing'] = df['is_printing'].astype(bool)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for predictive maintenance"""
        if df.empty:
            return df
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Temperature-related features
        df['temp_difference'] = df['hotend_temp'] - df['bed_temp']
        df['temp_ratio'] = df['hotend_temp'] / (df['bed_temp'] + 1e-6)
        
        # Vibration features
        df['total_vibration'] = df['vibration_x'] + df['vibration_y'] + df['vibration_z']
        df['vibration_ratio'] = df['vibration_x'] / (df['vibration_y'] + 1e-6)
        
        # Motor current features
        df['total_motor_current'] = (df['motor_current_x'] + df['motor_current_y'] + 
                                   df['motor_current_z'] + df['motor_current_extruder'])
        df['motor_current_ratio'] = df['motor_current_extruder'] / (df['total_motor_current'] + 1e-6)
        
        # Position-based features
        df['total_movement'] = df['position_x'] + df['position_y'] + df['position_z']
        df['movement_ratio'] = df['position_z'] / (df['total_movement'] + 1e-6)
        
        # Maintenance indicators
        df['maintenance_score'] = (df['belt_tension'] + (1 - df['nozzle_wear']) + 
                                 (1 - abs(df['bed_level'])) + (1 - df['extruder_clogging'])) / 4
        
        # Rolling statistics (if enough data)
        if len(df) > 10:
            window_size = min(10, len(df) // 2)
            
            # Rolling means
            for col in ['hotend_temp', 'bed_temp', 'total_vibration', 'total_motor_current']:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for predictive maintenance"""
        if df.empty:
            return df
        
        # Define thresholds for maintenance alerts
        temp_threshold = 250  # Max hotend temperature
        vibration_threshold = 0.8  # High vibration threshold
        current_threshold = 1.5  # High motor current threshold
        maintenance_threshold = 0.6  # Low maintenance score threshold
        
        # Create binary targets
        df['temp_alert'] = (df['hotend_temp'] > temp_threshold).astype(int)
        df['vibration_alert'] = (df['total_vibration'] > vibration_threshold).astype(int)
        df['current_alert'] = (df['total_motor_current'] > current_threshold).astype(int)
        df['maintenance_alert'] = (df['maintenance_score'] < maintenance_threshold).astype(int)
        
        # Create smarter alert: only mark as alert if 2 or more types fail
        df['any_alert'] = (
            df['temp_alert'].astype(int)
            + df['vibration_alert'].astype(int)
            + df['current_alert'].astype(int)
            + df['maintenance_alert'].astype(int)
        ) >= 2
        df['any_alert'] = df['any_alert'].astype(int)

        # Create time-to-failure estimate (simplified)
        df['time_to_maintenance'] = np.where(
            df['maintenance_score'] < 0.7,
            np.maximum(0, 100 - df['maintenance_score'] * 100),
            100
        )
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature columns for modeling"""
        if df.empty:
            return df, []
        
        # Select feature columns (exclude timestamp and target variables)
        exclude_cols = ['timestamp', 'temp_alert', 'vibration_alert', 'current_alert', 
                       'maintenance_alert', 'any_alert', 'time_to_maintenance']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        return df[feature_cols], feature_cols
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        if df.empty:
            return df
        
        # Select only numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fit:
            df_scaled = df.copy()
            df_scaled[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df_scaled = df.copy()
            df_scaled[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df_scaled
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create time sequences for LSTM models"""
        if len(df) < sequence_length:
            return np.array([]), np.array([])
        
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            sequence = df.iloc[i:i+sequence_length][self.feature_columns].values
            target = df.iloc[i+sequence_length]['any_alert']
            
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def preprocess_pipeline(self, filename: str, sequence_length: int = 10) -> Dict:
        """Complete preprocessing pipeline"""
        self.logger.info(f"Starting preprocessing pipeline for {filename}")
        
        # Load data
        df = self.load_json_data(filename)
        if df.empty:
            self.logger.error("No data loaded")
            return {}
        
        self.logger.info(f"Loaded {len(df)} records")
        
        # Clean data
        df = self.clean_data(df)
        self.logger.info(f"Cleaned data: {len(df)} records")
        
        # Engineer features
        df = self.engineer_features(df)
        self.logger.info("Feature engineering completed")
        
        # Create targets
        df = self.create_target_variables(df)
        self.logger.info("Target variables created")
        
        # Prepare features
        feature_df, feature_cols = self.prepare_features(df)
        self.logger.info(f"Prepared {len(feature_cols)} features")
        
        # Scale features
        scaled_features = self.scale_features(feature_df)
        
        # Create sequences for time series models
        X_sequences, y_sequences = self.create_sequences(df, sequence_length)
        
        # Prepare regular features for traditional ML
        X_regular = scaled_features.values
        y_regular = df['any_alert'].values
        
        return {
            'regular_features': X_regular,
            'regular_targets': y_regular,
            'sequences': X_sequences,
            'sequence_targets': y_sequences,
            'feature_names': feature_cols,
            'processed_df': df
        }

if __name__ == "__main__":
    # Example usage
    preprocessor = PrinterDataPreprocessor()
    
    # Test with simulated data
    import random
    from datetime import datetime, timedelta
    
    # Create sample data
sample_data = []
base_time = datetime.now()

for i in range(300):
    trigger_alert = random.random() < 0.5  # 50% alert, 50% healthy

    if trigger_alert:
        # ALERT CASE (class 1 expected)
        hotend = 270  # >250
        vibration = {'x_axis': 0.9, 'y_axis': 0.9, 'z_axis': 0.5}  # >0.8, >0.4
        motor_current = {'x_motor': 1.5, 'y_motor': 1.5, 'z_motor': 1.0, 'extruder': 1.6}  # >1.5
        maintenance = {
            'belt_tension': 0.3,  # <0.5
            'nozzle_wear': 0.9,
            'bed_level': 0.1,
            'extruder_clogging': 0.85
        }
    else:
        # HEALTHY CASE (class 0 expected)
        hotend = 220  # safe
        vibration = {'x_axis': 0.1, 'y_axis': 0.1, 'z_axis': 0.1}
        motor_current = {'x_motor': 1.0, 'y_motor': 1.0, 'z_motor': 0.5, 'extruder': 0.8}
        maintenance = {
            'belt_tension': 0.8,  # >0.5
            'nozzle_wear': 0.1,
            'bed_level': 0.0,
            'extruder_clogging': 0.2
        }

    sample_data.append({
        'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
        'temperature': {
            'hotend': hotend,
            'bed': 60 + random.uniform(-2, 2),
            'ambient': 25 + random.uniform(-3, 3)
        },
        'vibration': vibration,
        'motor_current': motor_current,
        'position': {
            'x': random.uniform(0, 200),
            'y': random.uniform(0, 200),
            'z': random.uniform(0, 200)
        },
        'print_status': {
            'is_printing': random.choice([True, False]),
            'layer_height': random.uniform(0.1, 0.3),
            'print_progress': random.uniform(0, 100),
            'filament_used': random.uniform(0, 100)
        },
        'maintenance_indicators': maintenance
    })


    # Save sample data
    with open('sample_printer_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Run preprocessing
    preprocessor = PrinterDataPreprocessor()

    result = preprocessor.preprocess_pipeline('sample_printer_data.json')
    
    if result:
        print(f"Preprocessing completed:")
        print(f"Regular features shape: {result['regular_features'].shape}")
        print(f"Sequences shape: {result['sequences'].shape}")
        print(f"Feature names: {result['feature_names']}") 