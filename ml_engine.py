"""
Machine Learning Engine for Power Grid Analysis
Handles all ML processing: training, prediction, and model management
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PowerGridMLEngine:
    """
    Complete ML engine for power grid analysis
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models
        self.stability_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.load_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        
    def prepare_data(self, data):
        """
        Prepare data for ML models
        
        Args:
            data (pd.DataFrame): Raw power grid data
            
        Returns:
            dict: Prepared datasets for different models
        """
        # Features for stability prediction
        stability_features = ['Voltage', 'Current', 'Frequency', 'Power_Factor', 'Load', 'Phase_Angle']
        
        # Features for load prediction (excluding load itself)
        load_features = ['Voltage', 'Current', 'Frequency', 'Power_Factor', 'Phase_Angle']
        
        # Features for anomaly detection
        anomaly_features = ['Voltage', 'Current', 'Frequency', 'Power_Factor', 'Load']
        
        prepared_data = {
            'stability': {
                'X': data[stability_features],
                'y': data['Stability'] if 'Stability' in data.columns else None
            },
            'load': {
                'X': data[load_features],
                'y': data['Load'] if 'Load' in data.columns else None
            },
            'anomaly': {
                'X': data[anomaly_features]
            }
        }
        
        return prepared_data
    
    def train_stability_model(self, X, y):
        """Train grid stability classification model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        self.stability_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.stability_model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results['stability'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'feature_importance': dict(zip(X.columns, self.stability_model.feature_importances_)),
            'predictions': y_pred,
            'actual': y_test
        }
        
        # Save model and scaler
        self.models['stability'] = self.stability_model
        self.scalers['stability'] = scaler
        
        return self.results['stability']
    
    def train_load_model(self, X, y):
        """Train load demand prediction model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        self.load_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.load_model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        self.results['load'] = {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'feature_importance': dict(zip(X.columns, self.load_model.feature_importances_)),
            'predictions': y_pred,
            'actual': y_test
        }
        
        # Save model and scaler
        self.models['load'] = self.load_model
        self.scalers['load'] = scaler
        
        return self.results['load']
    
    def train_anomaly_model(self, X):
        """Train anomaly detection model"""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        self.anomaly_model.fit(X_scaled)
        
        # Detect anomalies
        anomaly_labels = self.anomaly_model.predict(X_scaled)
        anomaly_scores = self.anomaly_model.decision_function(X_scaled)
        
        # Calculate metrics
        anomaly_percentage = (anomaly_labels == -1).sum() / len(anomaly_labels) * 100
        
        # Store results
        self.results['anomaly'] = {
            'anomaly_percentage': anomaly_percentage,
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'normal_count': (anomaly_labels == 1).sum(),
            'anomaly_count': (anomaly_labels == -1).sum()
        }
        
        # Save model and scaler
        self.models['anomaly'] = self.anomaly_model
        self.scalers['anomaly'] = scaler
        
        return self.results['anomaly']
    
    def train_all_models(self, data):
        """
        Train all ML models on the provided data
        
        Args:
            data (pd.DataFrame): Power grid dataset
            
        Returns:
            dict: All training results
        """
        print("🔄 Preparing data...")
        prepared_data = self.prepare_data(data)
        
        results = {}
        
        # Train stability model
        if prepared_data['stability']['y'] is not None:
            print("🔄 Training stability classification model...")
            results['stability'] = self.train_stability_model(
                prepared_data['stability']['X'], 
                prepared_data['stability']['y']
            )
        
        # Train load prediction model
        if prepared_data['load']['y'] is not None:
            print("🔄 Training load prediction model...")
            results['load'] = self.train_load_model(
                prepared_data['load']['X'], 
                prepared_data['load']['y']
            )
        
        # Train anomaly detection model
        print("🔄 Training anomaly detection model...")
        results['anomaly'] = self.train_anomaly_model(prepared_data['anomaly']['X'])
        
        print("✅ All models trained successfully!")
        return results
    
    def save_models(self):
        """Save all trained models and scalers"""
        for model_name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            
        for scaler_name, scaler in self.scalers.items():
            scaler_path = os.path.join(self.models_dir, f'{scaler_name}_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            
        print("💾 Models and scalers saved successfully!")
    
    def load_models(self):
        """Load saved models and scalers"""
        model_files = ['stability_model.pkl', 'load_model.pkl', 'anomaly_model.pkl']
        scaler_files = ['stability_scaler.pkl', 'load_scaler.pkl', 'anomaly_scaler.pkl']
        
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('_model.pkl', '')
                self.models[model_name] = joblib.load(model_path)
        
        for scaler_file in scaler_files:
            scaler_path = os.path.join(self.models_dir, scaler_file)
            if os.path.exists(scaler_path):
                scaler_name = scaler_file.replace('_scaler.pkl', '')
                self.scalers[scaler_name] = joblib.load(scaler_path)
    
    def predict_stability(self, X):
        """Predict grid stability for new data"""
        if 'stability' in self.models and 'stability' in self.scalers:
            X_scaled = self.scalers['stability'].transform(X)
            predictions = self.models['stability'].predict(X_scaled)
            probabilities = self.models['stability'].predict_proba(X_scaled)
            return predictions, probabilities
        return None, None
    
    def predict_load(self, X):
        """Predict load demand for new data"""
        if 'load' in self.models and 'load' in self.scalers:
            X_scaled = self.scalers['load'].transform(X)
            predictions = self.models['load'].predict(X_scaled)
            return predictions
        return None
    
    def detect_anomalies(self, X):
        """Detect anomalies in new data"""
        if 'anomaly' in self.models and 'anomaly' in self.scalers:
            X_scaled = self.scalers['anomaly'].transform(X)
            anomaly_labels = self.models['anomaly'].predict(X_scaled)
            anomaly_scores = self.models['anomaly'].decision_function(X_scaled)
            return anomaly_labels, anomaly_scores
        return None, None
    
    def generate_insights(self, data):
        """
        Generate automated insights from the analysis
        
        Args:
            data (pd.DataFrame): Original dataset
            
        Returns:
            list: List of insight strings
        """
        insights = []
        
        if 'stability' in self.results:
            stability_acc = self.results['stability']['accuracy']
            insights.append(f"🎯 Grid stability prediction achieved {stability_acc:.1%} accuracy")
            
            # Feature importance insights
            feature_imp = self.results['stability']['feature_importance']
            top_feature = max(feature_imp, key=feature_imp.get)
            insights.append(f"⚡ {top_feature} is the most critical factor for grid stability")
        
        if 'load' in self.results:
            r2 = self.results['load']['r2_score']
            rmse = self.results['load']['rmse']
            insights.append(f"📊 Load prediction model explains {r2:.1%} of demand variance (RMSE: {rmse:.2f} kW)")
        
        if 'anomaly' in self.results:
            anomaly_pct = self.results['anomaly']['anomaly_percentage']
            insights.append(f"🚨 Detected {anomaly_pct:.1f}% anomalous readings in the grid data")
            
            if anomaly_pct > 10:
                insights.append("⚠️ High anomaly rate detected - investigate grid equipment")
            elif anomaly_pct < 2:
                insights.append("✅ Grid operating within normal parameters")
        
        # Data-driven insights
        if 'Load' in data.columns:
            avg_load = data['Load'].mean()
            max_load = data['Load'].max()
            insights.append(f"⚡ Average grid load: {avg_load:.1f} kW (Peak: {max_load:.1f} kW)")
        
        if 'Stability' in data.columns:
            stability_rate = data['Stability'].mean()
            insights.append(f"🔒 Grid stability rate: {stability_rate:.1%}")
        
        return insights

if __name__ == "__main__":
    # Test the ML engine
    from data_generator import generate_power_grid_data
    
    # Generate test data
    data = generate_power_grid_data(1000)
    
    # Initialize and train models
    ml_engine = PowerGridMLEngine()
    results = ml_engine.train_all_models(data)
    
    # Save models
    ml_engine.save_models()
    
    print("ML Engine test completed successfully!")