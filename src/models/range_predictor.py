"""
EV Range Prediction Models
Advanced ML models for predicting EV driving range based on multiple factors
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RangePredictionModel:
    """
    Advanced ML model for EV range prediction incorporating:
    - Battery state and health
    - Weather conditions
    - Driving patterns
    - Route characteristics
    - Vehicle efficiency factors
    """
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'linear': LinearRegression()
        }
        
        self.scalers = {
            'numerical': StandardScaler(),
            'target': StandardScaler()
        }
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic training data based on real-world EV patterns
        This simulates the kind of data you'd collect from actual EV fleets
        """
        
        np.random.seed(42)
        
        # Battery and vehicle characteristics
        battery_capacity = np.random.normal(75, 10, n_samples)  # kWh
        battery_soh = np.random.beta(9, 1, n_samples)  # State of Health (0.7-1.0)
        battery_soc_start = np.random.uniform(0.2, 1.0, n_samples)  # Starting SOC
        vehicle_efficiency = np.random.normal(4.5, 0.8, n_samples)  # km/kWh
        
        # Weather conditions
        temperature = np.random.normal(20, 15, n_samples)  # Celsius
        humidity = np.random.uniform(30, 90, n_samples)  # Percentage
        wind_speed = np.random.exponential(10, n_samples)  # km/h
        precipitation = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Boolean
        
        # Driving conditions
        avg_speed = np.random.gamma(2, 15, n_samples)  # km/h
        traffic_density = np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.3, 0.5, 0.2])
        route_type = np.random.choice(['city', 'highway', 'mixed'], n_samples, p=[0.4, 0.3, 0.3])
        elevation_change = np.random.normal(0, 100, n_samples)  # meters
        
        # Driver behavior
        driving_style = np.random.choice(['eco', 'normal', 'aggressive'], n_samples, p=[0.2, 0.6, 0.2])
        ac_usage = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Boolean
        
        # Calculate realistic range based on these factors
        base_range = battery_capacity * battery_soh * battery_soc_start * vehicle_efficiency
        
        # Apply environmental factors
        temp_factor = 1 - 0.002 * abs(temperature - 20)  # Optimal at 20°C
        weather_factor = 1 - 0.1 * precipitation - 0.001 * wind_speed
        
        # Apply driving factors
        speed_factor = np.where(avg_speed < 60, 1.0, 1 - 0.005 * (avg_speed - 60))
        traffic_factor = {'low': 1.0, 'medium': 0.9, 'high': 0.8}
        traffic_multiplier = np.array([traffic_factor[t] for t in traffic_density])
        
        route_factor = {'city': 0.85, 'highway': 1.1, 'mixed': 1.0}
        route_multiplier = np.array([route_factor[r] for r in route_type])
        
        elevation_factor = 1 - 0.0001 * abs(elevation_change)
        
        driving_factor = {'eco': 1.2, 'normal': 1.0, 'aggressive': 0.8}
        driving_multiplier = np.array([driving_factor[d] for d in driving_style])
        
        ac_factor = np.where(ac_usage, 0.9, 1.0)
        
        # Calculate final range
        range_km = (base_range * temp_factor * weather_factor * speed_factor * 
                   traffic_multiplier * route_multiplier * elevation_factor * 
                   driving_multiplier * ac_factor)
        
        # Add some realistic noise
        range_km += np.random.normal(0, 5, n_samples)
        range_km = np.clip(range_km, 50, 600)  # Realistic range limits
        
        # Create DataFrame
        data = pd.DataFrame({
            'battery_capacity_kwh': battery_capacity,
            'battery_soh': battery_soh,
            'battery_soc_start': battery_soc_start,
            'vehicle_efficiency_km_kwh': vehicle_efficiency,
            'temperature_c': temperature,
            'humidity_percent': humidity,
            'wind_speed_kmh': wind_speed,
            'precipitation': precipitation,
            'avg_speed_kmh': avg_speed,
            'traffic_density': traffic_density,
            'route_type': route_type,
            'elevation_change_m': elevation_change,
            'driving_style': driving_style,
            'ac_usage': ac_usage,
            'range_km': range_km
        })
        
        return data
    
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess data for training/prediction"""
        
        data_processed = data.copy()
        
        # Handle categorical variables
        categorical_cols = ['traffic_density', 'route_type', 'driving_style']
        
        for col in categorical_cols:
            if col in data_processed.columns:
                if is_training:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        data_processed[col] = self.encoders[col].fit_transform(data_processed[col])
                    else:
                        data_processed[col] = self.encoders[col].transform(data_processed[col])
                else:
                    if col in self.encoders:
                        data_processed[col] = self.encoders[col].transform(data_processed[col])
        
        return data_processed
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the range prediction models"""
        
        # Preprocess data
        data_processed = self.preprocess_data(data, is_training=True)
        
        # Separate features and target
        feature_cols = [col for col in data_processed.columns if col != 'range_km']
        X = data_processed[feature_cols]
        y = data_processed['range_km']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scalers['numerical'].fit_transform(X_train)
        X_test_scaled = self.scalers['numerical'].transform(X_test)
        
        # Train models
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name} model...")
            
            if name == 'linear':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
        
        self.is_trained = True
        self.feature_columns = feature_cols
        
        return results
    
    def predict_range(self, input_data: Dict) -> Dict:
        """
        Predict range for given input conditions
        
        Args:
            input_data: Dictionary with vehicle and environmental parameters
            
        Returns:
            Prediction results from different models
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess
        input_processed = self.preprocess_data(input_df, is_training=False)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in input_processed.columns:
                input_processed[col] = 0  # Default value
        
        input_processed = input_processed[self.feature_columns]
        
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'linear':
                X_scaled = self.scalers['numerical'].transform(input_processed)
                pred = model.predict(X_scaled)[0]
            else:
                pred = model.predict(input_processed)[0]
            
            predictions[name] = max(0, pred)  # Ensure non-negative
        
        # Ensemble prediction (weighted average)
        weights = {'random_forest': 0.4, 'gradient_boost': 0.4, 'linear': 0.2}
        ensemble_pred = sum(predictions[name] * weights[name] for name in predictions.keys())
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def optimize_route_efficiency(self, start_range: float, destination_distance: float,
                                available_charging_stations: List[Dict]) -> Dict:
        """
        Optimize route for maximum efficiency and range confidence
        
        Args:
            start_range: Current predicted range
            destination_distance: Distance to destination
            available_charging_stations: List of charging stations with distances
            
        Returns:
            Route optimization recommendations
        """
        
        safety_margin = 0.2  # 20% safety margin
        required_range = destination_distance * (1 + safety_margin)
        
        if start_range >= required_range:
            return {
                'can_reach_destination': True,
                'charging_needed': False,
                'recommended_stations': [],
                'confidence': 'High',
                'safety_margin_km': start_range - destination_distance
            }
        
        # Find optimal charging stations
        suitable_stations = []
        for station in available_charging_stations:
            if station['distance_km'] < start_range * 0.8:  # Reachable with margin
                suitable_stations.append(station)
        
        if not suitable_stations:
            return {
                'can_reach_destination': False,
                'charging_needed': True,
                'recommended_stations': [],
                'confidence': 'Low',
                'issue': 'No reachable charging stations'
            }
        
        # Sort by distance and choose optimal
        suitable_stations.sort(key=lambda x: x['distance_km'])
        recommended = suitable_stations[:2]  # Top 2 options
        
        return {
            'can_reach_destination': True,
            'charging_needed': True,
            'recommended_stations': recommended,
            'confidence': 'Medium',
            'charging_time_estimate': '30-45 minutes'
        }
    
    def get_range_factors_analysis(self, input_data: Dict) -> Dict:
        """Analyze which factors most impact the range prediction"""
        
        if not self.is_trained or 'random_forest' not in self.feature_importance:
            return {}
        
        importance = self.feature_importance['random_forest']
        
        # Get top factors
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Map technical names to user-friendly names
        factor_names = {
            'battery_capacity_kwh': 'Battery Capacity',
            'battery_soh': 'Battery Health',
            'battery_soc_start': 'Starting Charge Level',
            'vehicle_efficiency_km_kwh': 'Vehicle Efficiency',
            'temperature_c': 'Temperature',
            'avg_speed_kmh': 'Average Speed',
            'driving_style': 'Driving Style',
            'route_type': 'Route Type',
            'ac_usage': 'Air Conditioning',
            'traffic_density': 'Traffic Conditions'
        }
        
        analysis = {
            'top_factors': [
                {
                    'factor': factor_names.get(factor, factor),
                    'importance': importance * 100,
                    'current_value': input_data.get(factor, 'N/A')
                }
                for factor, importance in sorted_importance[:5]
            ],
            'recommendations': []
        }
        
        # Generate recommendations based on current values
        if input_data.get('temperature_c', 20) < 0:
            analysis['recommendations'].append("Cold weather detected - consider preheating battery")
        
        if input_data.get('avg_speed_kmh', 50) > 80:
            analysis['recommendations'].append("High speed reduces efficiency - consider moderate speeds")
        
        if input_data.get('driving_style') == 'aggressive':
            analysis['recommendations'].append("Eco driving mode can improve range by 15-20%")
        
        return analysis
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_importance = model_data['feature_importance']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = True
