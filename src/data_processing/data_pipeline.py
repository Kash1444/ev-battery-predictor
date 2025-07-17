"""
Data Processing Pipeline for EV Battery and Range Data
Handles data collection, cleaning, feature engineering, and preparation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EVDataProcessor:
    """
    Comprehensive data processing pipeline for EV battery and driving data
    """
    
    def __init__(self):
        self.processed_data = None
        self.feature_stats = {}
        
    def generate_realistic_ev_dataset(self, n_vehicles: int = 100, 
                                    days_per_vehicle: int = 365) -> pd.DataFrame:
        """
        Generate a realistic EV dataset simulating real-world fleet data
        
        Args:
            n_vehicles: Number of vehicles to simulate
            days_per_vehicle: Number of days of data per vehicle
            
        Returns:
            Comprehensive EV dataset with battery and driving data
        """
        
        np.random.seed(42)
        all_data = []
        
        for vehicle_id in range(n_vehicles):
            # Vehicle characteristics
            vehicle_type = np.random.choice(['compact', 'sedan', 'suv'], p=[0.3, 0.5, 0.2])
            battery_capacity = {
                'compact': np.random.normal(40, 5),
                'sedan': np.random.normal(75, 8),
                'suv': np.random.normal(100, 10)
            }[vehicle_type]
            
            # Owner characteristics
            driver_type = np.random.choice(['conservative', 'normal', 'aggressive'], p=[0.2, 0.6, 0.2])
            annual_mileage = np.random.normal(15000, 5000)
            daily_distance = annual_mileage / 365
            
            # Generate daily data for this vehicle
            for day in range(days_per_vehicle):
                date = datetime.now() - timedelta(days=days_per_vehicle - day)
                
                # Seasonal effects
                temp_base = 20 + 15 * np.sin(2 * np.pi * day / 365)
                temperature = np.random.normal(temp_base, 5)
                
                # Battery aging
                age_days = day
                initial_soh = 1.0
                aging_rate = 0.00008  # ~8% per year
                current_soh = max(0.7, initial_soh - aging_rate * age_days)
                
                # Daily driving pattern
                trips_per_day = np.random.poisson(2.5)
                
                for trip in range(max(1, trips_per_day)):
                    trip_distance = np.random.gamma(2, daily_distance / max(1, trips_per_day))
                    trip_distance = min(trip_distance, 200)  # Max single trip
                    
                    # Trip characteristics
                    hour = np.random.choice(range(24), p=self._get_hourly_distribution())
                    route_type = self._determine_route_type(hour, trip_distance)
                    traffic = self._determine_traffic(hour, route_type)
                    
                    # Weather conditions
                    humidity = np.random.uniform(30, 90)
                    wind_speed = np.random.exponential(8)
                    precipitation = np.random.choice([0, 1], p=[0.8, 0.2])
                    
                    # Driving conditions
                    avg_speed = self._calculate_avg_speed(route_type, traffic, hour)
                    elevation_change = np.random.normal(0, 50)
                    
                    # Driver behavior
                    driving_aggressiveness = {
                        'conservative': np.random.uniform(0.7, 0.9),
                        'normal': np.random.uniform(0.8, 1.2),
                        'aggressive': np.random.uniform(1.1, 1.5)
                    }[driver_type]
                    
                    # Vehicle usage
                    ac_usage = 1 if abs(temperature - 22) > 8 else np.random.choice([0, 1], p=[0.3, 0.7])
                    heating_usage = 1 if temperature < 10 else 0
                    
                    # Battery state
                    soc_start = np.random.uniform(0.2, 0.95)
                    
                    # Energy consumption calculation
                    base_consumption = self._calculate_energy_consumption(
                        vehicle_type, trip_distance, avg_speed, temperature,
                        elevation_change, driving_aggressiveness, ac_usage, heating_usage
                    )
                    
                    # Range calculation
                    available_energy = battery_capacity * current_soh * soc_start
                    theoretical_range = available_energy / (base_consumption / trip_distance)
                    
                    # Add realistic noise and constraints
                    actual_range = theoretical_range * np.random.normal(1.0, 0.1)
                    actual_range = np.clip(actual_range, 50, 600)
                    
                    # Cycle count (simplified)
                    daily_cycles = trip / max(1, trips_per_day)
                    total_cycles = age_days * 1.2  # Approximate total cycles
                    
                    trip_data = {
                        'vehicle_id': vehicle_id,
                        'date': date.strftime('%Y-%m-%d'),
                        'trip_id': trip,
                        'vehicle_type': vehicle_type,
                        'battery_capacity_kwh': battery_capacity,
                        'battery_soh': current_soh,
                        'battery_soc_start': soc_start,
                        'cycle_count': total_cycles,
                        'age_days': age_days,
                        'temperature_c': temperature,
                        'humidity_percent': humidity,
                        'wind_speed_kmh': wind_speed,
                        'precipitation': precipitation,
                        'trip_distance_km': trip_distance,
                        'avg_speed_kmh': avg_speed,
                        'route_type': route_type,
                        'traffic_density': traffic,
                        'elevation_change_m': elevation_change,
                        'hour_of_day': hour,
                        'driver_type': driver_type,
                        'driving_aggressiveness': driving_aggressiveness,
                        'ac_usage': ac_usage,
                        'heating_usage': heating_usage,
                        'energy_consumption_kwh': base_consumption,
                        'predicted_range_km': actual_range,
                        'actual_range_km': actual_range * np.random.normal(1.0, 0.05)  # Slight measurement noise
                    }
                    
                    all_data.append(trip_data)
        
        return pd.DataFrame(all_data)
    
    def _get_hourly_distribution(self) -> List[float]:
        """Get realistic hourly distribution of trips"""
        # Peak hours: 7-9 AM, 5-7 PM
        probs = [0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.06, 0.12, 0.08, 0.04,
                 0.03, 0.04, 0.05, 0.04, 0.03, 0.04, 0.06, 0.12, 0.08, 0.05,
                 0.04, 0.03, 0.02, 0.01]
        # Normalize to ensure probabilities sum to 1.0
        probs = np.array(probs)
        probs = probs / probs.sum()
        return probs.tolist()
    
    def _determine_route_type(self, hour: int, distance: float) -> str:
        """Determine route type based on time and distance"""
        if distance < 5:
            return 'city'
        elif distance > 50:
            return 'highway'
        elif 6 <= hour <= 9 or 17 <= hour <= 19:
            return 'mixed'  # Commute hours
        else:
            return np.random.choice(['city', 'mixed'], p=[0.7, 0.3])
    
    def _determine_traffic(self, hour: int, route_type: str) -> str:
        """Determine traffic conditions"""
        if route_type == 'highway':
            if 6 <= hour <= 9 or 17 <= hour <= 19:
                return np.random.choice(['medium', 'high'], p=[0.4, 0.6])
            else:
                return np.random.choice(['low', 'medium'], p=[0.7, 0.3])
        else:
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return np.random.choice(['medium', 'high'], p=[0.5, 0.5])
            else:
                return np.random.choice(['low', 'medium'], p=[0.6, 0.4])
    
    def _calculate_avg_speed(self, route_type: str, traffic: str, hour: int) -> float:
        """Calculate average speed based on conditions"""
        base_speeds = {
            'city': 35,
            'mixed': 55,
            'highway': 85
        }
        
        traffic_factors = {
            'low': 1.0,
            'medium': 0.8,
            'high': 0.6
        }
        
        base_speed = base_speeds[route_type]
        traffic_factor = traffic_factors[traffic]
        
        # Night time slightly higher speeds
        time_factor = 1.1 if 22 <= hour or hour <= 5 else 1.0
        
        speed = base_speed * traffic_factor * time_factor
        return np.random.normal(speed, speed * 0.1)
    
    def _calculate_energy_consumption(self, vehicle_type: str, distance: float,
                                    speed: float, temperature: float,
                                    elevation: float, aggressiveness: float,
                                    ac_usage: int, heating_usage: int) -> float:
        """Calculate energy consumption for trip"""
        
        # Base consumption rates (kWh/100km)
        base_rates = {
            'compact': 16,
            'sedan': 20,
            'suv': 25
        }
        
        base_rate = base_rates[vehicle_type]
        
        # Speed efficiency curve (optimal around 50-60 kmh)
        if speed < 30:
            speed_factor = 1.2
        elif speed > 100:
            speed_factor = 1.3 + (speed - 100) * 0.01
        else:
            speed_factor = 1.0 + abs(speed - 55) * 0.002
        
        # Temperature effects
        temp_factor = 1.0 + abs(temperature - 20) * 0.015
        
        # Elevation effects
        elevation_factor = 1.0 + max(0, elevation) * 0.0001
        
        # Accessories
        accessory_factor = 1.0 + ac_usage * 0.15 + heating_usage * 0.25
        
        # Driving style
        style_factor = aggressiveness
        
        total_consumption = (base_rate * distance / 100 * speed_factor * 
                           temp_factor * elevation_factor * accessory_factor * style_factor)
        
        return max(0.5, total_consumption)  # Minimum consumption
    
    def clean_and_validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        
        print("Cleaning and validating data...")
        
        # Remove duplicate records
        initial_count = len(data)
        data = data.drop_duplicates()
        print(f"Removed {initial_count - len(data)} duplicate records")
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Remove outliers (beyond 3 standard deviations)
        for col in ['predicted_range_km', 'energy_consumption_kwh', 'avg_speed_kmh']:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                print(f"Removed {outliers_count} outliers from {col}")
        
        # Validate ranges
        data = data[data['battery_soh'] >= 0.5]  # Minimum viable SOH
        data = data[data['battery_soc_start'] >= 0.05]  # Minimum SOC
        data = data[data['predicted_range_km'] >= 20]  # Minimum range
        data = data[data['avg_speed_kmh'] >= 5]  # Minimum speed
        
        print(f"Final dataset size: {len(data)} records")
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better model performance"""
        
        print("Engineering features...")
        
        # Time-based features
        data['date'] = pd.to_datetime(data['date'])
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data['season'] = data['month'].map({12: 'winter', 1: 'winter', 2: 'winter',
                                           3: 'spring', 4: 'spring', 5: 'spring',
                                           6: 'summer', 7: 'summer', 8: 'summer',
                                           9: 'autumn', 10: 'autumn', 11: 'autumn'})
        
        # Battery features
        data['usable_capacity'] = data['battery_capacity_kwh'] * data['battery_soh']
        data['available_energy'] = data['usable_capacity'] * data['battery_soc_start']
        data['battery_age_years'] = data['age_days'] / 365
        data['cycles_per_day'] = data['cycle_count'] / (data['age_days'] + 1)
        
        # Environmental features
        data['temp_category'] = pd.cut(data['temperature_c'], 
                                     bins=[-np.inf, 0, 10, 25, 35, np.inf],
                                     labels=['very_cold', 'cold', 'optimal', 'warm', 'hot'])
        
        data['weather_severity'] = (data['precipitation'] * 0.3 + 
                                   (data['wind_speed_kmh'] > 20).astype(int) * 0.2 +
                                   (abs(data['temperature_c'] - 20) > 15).astype(int) * 0.5)
        
        # Driving features
        data['speed_efficiency'] = np.where(
            (data['avg_speed_kmh'] >= 45) & (data['avg_speed_kmh'] <= 65), 1.0,
            1.0 - abs(data['avg_speed_kmh'] - 55) * 0.005
        )
        
        data['trip_category'] = pd.cut(data['trip_distance_km'],
                                     bins=[0, 5, 20, 50, np.inf],
                                     labels=['short', 'medium', 'long', 'very_long'])
        
        # Efficiency metrics
        data['energy_efficiency'] = data['trip_distance_km'] / (data['energy_consumption_kwh'] + 0.001)
        data['range_efficiency'] = data['predicted_range_km'] / (data['available_energy'] + 0.001)
        
        # Interaction features
        data['temp_speed_interaction'] = data['temperature_c'] * data['avg_speed_kmh'] / 1000
        data['soh_age_interaction'] = data['battery_soh'] * data['battery_age_years']
        data['traffic_speed_ratio'] = data['avg_speed_kmh'] / data.groupby('traffic_density')['avg_speed_kmh'].transform('mean')
        
        # Peak hour indicator
        data['is_peak_hour'] = ((data['hour_of_day'].between(7, 9)) | 
                               (data['hour_of_day'].between(17, 19))).astype(int)
        
        # Weekend indicator
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        print(f"Added features. New dataset shape: {data.shape}")
        return data
    
    def prepare_model_data(self, data: pd.DataFrame, target_col: str = 'predicted_range_km') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training"""
        
        # Select features for modeling
        feature_columns = [
            # Battery features
            'battery_capacity_kwh', 'battery_soh', 'battery_soc_start', 
            'usable_capacity', 'available_energy', 'battery_age_years',
            
            # Environmental features
            'temperature_c', 'humidity_percent', 'wind_speed_kmh', 'precipitation',
            'weather_severity',
            
            # Driving features
            'trip_distance_km', 'avg_speed_kmh', 'elevation_change_m',
            'driving_aggressiveness', 'speed_efficiency',
            
            # Vehicle features
            'ac_usage', 'heating_usage',
            
            # Temporal features
            'hour_of_day', 'day_of_week', 'month', 'is_peak_hour', 'is_weekend',
            
            # Categorical features (will be encoded)
            'vehicle_type', 'route_type', 'traffic_density', 'driver_type',
            'temp_category', 'trip_category', 'season',
            
            # Interaction features
            'temp_speed_interaction', 'soh_age_interaction', 'traffic_speed_ratio'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        X = data[available_features].copy()
        y = data[target_col].copy()
        
        # Store feature statistics
        self.feature_stats = {
            'numeric_features': X.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': X.select_dtypes(include=['object', 'category']).columns.tolist(),
            'feature_means': X.select_dtypes(include=[np.number]).mean().to_dict(),
            'feature_stds': X.select_dtypes(include=[np.number]).std().to_dict()
        }
        
        return X, y
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive data summary"""
        
        summary = {
            'basic_stats': {
                'total_records': len(data),
                'unique_vehicles': data['vehicle_id'].nunique() if 'vehicle_id' in data.columns else 'N/A',
                'date_range': f"{data['date'].min()} to {data['date'].max()}" if 'date' in data.columns else 'N/A',
                'total_distance_km': data['trip_distance_km'].sum() if 'trip_distance_km' in data.columns else 'N/A'
            },
            
            'battery_stats': {
                'avg_soh': data['battery_soh'].mean() if 'battery_soh' in data.columns else 'N/A',
                'avg_capacity': data['battery_capacity_kwh'].mean() if 'battery_capacity_kwh' in data.columns else 'N/A',
                'avg_cycles': data['cycle_count'].mean() if 'cycle_count' in data.columns else 'N/A'
            },
            
            'performance_stats': {
                'avg_range_km': data['predicted_range_km'].mean() if 'predicted_range_km' in data.columns else 'N/A',
                'avg_consumption': data['energy_consumption_kwh'].mean() if 'energy_consumption_kwh' in data.columns else 'N/A',
                'avg_efficiency': data['energy_efficiency'].mean() if 'energy_efficiency' in data.columns else 'N/A'
            },
            
            'environmental_stats': {
                'avg_temperature': data['temperature_c'].mean() if 'temperature_c' in data.columns else 'N/A',
                'precipitation_days': data['precipitation'].sum() if 'precipitation' in data.columns else 'N/A',
                'avg_speed': data['avg_speed_kmh'].mean() if 'avg_speed_kmh' in data.columns else 'N/A'
            }
        }
        
        return summary
