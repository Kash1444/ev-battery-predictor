"""
Utility functions for EV Battery Predictor
Common functions and helpers used across the application
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EVVisualization:
    """Visualization utilities for EV battery and range data"""
    
    @staticmethod
    def plot_battery_health_timeline(simulation_data: pd.DataFrame) -> go.Figure:
        """Create battery health timeline visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Battery Health Over Time', 'Remaining Capacity', 
                          'Cycle Count Impact', 'Temperature Effects'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # SOH timeline
        fig.add_trace(
            go.Scatter(x=simulation_data['month'], y=simulation_data['soh'],
                      name='State of Health', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # Capacity timeline
        fig.add_trace(
            go.Scatter(x=simulation_data['month'], y=simulation_data['capacity_kwh'],
                      name='Capacity (kWh)', line=dict(color='green', width=3)),
            row=1, col=2
        )
        
        # Cycle count vs SOH
        fig.add_trace(
            go.Scatter(x=simulation_data['cycle_count'], y=simulation_data['soh'],
                      mode='markers', name='SOH vs Cycles',
                      marker=dict(color='red', size=6)),
            row=2, col=1
        )
        
        # Add cycle trend line
        fig.add_trace(
            go.Scatter(x=simulation_data['cycle_count'], 
                      y=simulation_data['cycle_count'].rolling(10).mean(),
                      name='Cycle Trend', line=dict(color='orange', dash='dash')),
            row=2, col=1, secondary_y=True
        )
        
        # Temperature distribution
        fig.add_trace(
            go.Histogram(x=simulation_data['temperature'], name='Temperature Distribution',
                        marker=dict(color='purple', opacity=0.7)),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Battery Digital Twin Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_range_prediction_analysis(actual_range: List[float], 
                                     predicted_range: List[float],
                                     factors: Dict) -> go.Figure:
        """Create range prediction analysis visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Accuracy', 'Range Distribution', 
                          'Factor Importance', 'Error Analysis'),
        )
        
        # Prediction accuracy scatter
        fig.add_trace(
            go.Scatter(x=actual_range, y=predicted_range, mode='markers',
                      name='Predictions', marker=dict(color='blue', size=6)),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_range, max_range = min(actual_range + predicted_range), max(actual_range + predicted_range)
        fig.add_trace(
            go.Scatter(x=[min_range, max_range], y=[min_range, max_range],
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Range distributions
        fig.add_trace(
            go.Histogram(x=actual_range, name='Actual Range', 
                        marker=dict(color='blue', opacity=0.7)),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=predicted_range, name='Predicted Range',
                        marker=dict(color='green', opacity=0.7)),
            row=1, col=2
        )
        
        # Factor importance
        if factors:
            factor_names = list(factors.keys())
            factor_values = list(factors.values())
            
            fig.add_trace(
                go.Bar(x=factor_values, y=factor_names, orientation='h',
                      name='Factor Importance', marker=dict(color='orange')),
                row=2, col=1
            )
        
        # Error analysis
        errors = np.array(predicted_range) - np.array(actual_range)
        fig.add_trace(
            go.Histogram(x=errors, name='Prediction Errors',
                        marker=dict(color='red', opacity=0.7)),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Range Prediction Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_charging_optimization(charging_strategy: Dict) -> go.Figure:
        """Visualize charging optimization recommendations"""
        
        # Simulate charging curve
        time_hours = np.linspace(0, charging_strategy.get('estimated_time_hours', 2), 100)
        
        # Typical charging curve (fast initial, slower near full)
        soc_curve = 1 - np.exp(-3 * time_hours / charging_strategy.get('estimated_time_hours', 2))
        soc_curve = np.clip(soc_curve, 0, charging_strategy.get('target_soc', 0.8))
        
        fig = go.Figure()
        
        # Charging curve
        fig.add_trace(
            go.Scatter(x=time_hours, y=soc_curve * 100,
                      mode='lines', name='Optimal Charging Curve',
                      line=dict(color='green', width=3))
        )
        
        # Target SOC line
        target_soc = charging_strategy.get('target_soc', 0.8) * 100
        fig.add_hline(y=target_soc, line_dash="dash", line_color="red",
                     annotation_text=f"Target SOC: {target_soc:.0f}%")
        
        # Charging rate zones
        if charging_strategy.get('temperature_compensation', False):
            fig.add_vrect(x0=0, x1=charging_strategy.get('estimated_time_hours', 2) * 0.3,
                         fillcolor="yellow", opacity=0.2,
                         annotation_text="Temperature Compensation")
        
        fig.update_layout(
            title="Optimized Charging Strategy",
            xaxis_title="Time (hours)",
            yaxis_title="State of Charge (%)",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_dashboard_summary(battery_report: Dict, range_prediction: Dict) -> go.Figure:
        """Create comprehensive dashboard summary"""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Battery Health', 'Range Prediction', 'Charging Status',
                          'Temperature Impact', 'Usage Efficiency', 'Recommendations'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]]
        )
        
        # Battery health gauge
        soh_value = battery_report.get('state_of_health', 0.8) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=soh_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Battery Health (%)"},
                delta={'reference': 90},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Range prediction gauge
        range_value = range_prediction.get('ensemble', 300)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=range_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Range (km)"},
                gauge={
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 150], 'color': "red"},
                        {'range': [150, 300], 'color': "yellow"},
                        {'range': [300, 500], 'color': "lightgreen"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Charging status
        rul_days = battery_report.get('remaining_useful_life_days', 1000)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=rul_days,
                title={'text': "Remaining Life (days)"},
                delta={'reference': 1095, 'relative': True, 'position': "top"}
            ),
            row=1, col=3
        )
        
        return fig

class EVMetrics:
    """Performance metrics and calculations for EV systems"""
    
    @staticmethod
    def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive model performance metrics"""
        
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Accuracy within tolerance
        tolerance_5 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.05) * 100
        tolerance_10 = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.10) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'accuracy_5_percent': tolerance_5,
            'accuracy_10_percent': tolerance_10
        }
    
    @staticmethod
    def calculate_battery_efficiency(energy_consumed: float, distance: float,
                                   battery_capacity: float, soh: float) -> Dict:
        """Calculate battery and energy efficiency metrics"""
        
        efficiency_km_kwh = distance / energy_consumed if energy_consumed > 0 else 0
        efficiency_wh_km = (energy_consumed * 1000) / distance if distance > 0 else 0
        
        theoretical_max_range = battery_capacity * soh * efficiency_km_kwh
        
        return {
            'efficiency_km_kwh': efficiency_km_kwh,
            'efficiency_wh_km': efficiency_wh_km,
            'theoretical_max_range': theoretical_max_range,
            'energy_utilization': (energy_consumed / (battery_capacity * soh)) * 100 if battery_capacity * soh > 0 else 0
        }
    
    @staticmethod
    def calculate_environmental_impact(distance: float, efficiency: float,
                                     grid_carbon_intensity: float = 500) -> Dict:
        """Calculate environmental impact metrics"""
        
        energy_consumed = distance / efficiency if efficiency > 0 else 0
        co2_emissions_kg = (energy_consumed * grid_carbon_intensity) / 1000
        
        # Compare with ICE vehicle (average 120g CO2/km)
        ice_emissions_kg = distance * 0.12
        co2_savings_kg = ice_emissions_kg - co2_emissions_kg
        
        return {
            'energy_consumed_kwh': energy_consumed,
            'co2_emissions_kg': co2_emissions_kg,
            'ice_equivalent_emissions_kg': ice_emissions_kg,
            'co2_savings_kg': co2_savings_kg,
            'emissions_reduction_percent': (co2_savings_kg / ice_emissions_kg) * 100 if ice_emissions_kg > 0 else 0
        }

class EVUtils:
    """General utility functions for EV analysis"""
    
    @staticmethod
    def convert_temperature(temp: float, from_unit: str = 'C', to_unit: str = 'F') -> float:
        """Convert temperature between Celsius and Fahrenheit"""
        if from_unit == 'C' and to_unit == 'F':
            return (temp * 9/5) + 32
        elif from_unit == 'F' and to_unit == 'C':
            return (temp - 32) * 5/9
        else:
            return temp
    
    @staticmethod
    def estimate_charging_time(current_soc: float, target_soc: float,
                             battery_capacity: float, charging_power: float) -> float:
        """Estimate charging time in hours"""
        
        energy_needed = (target_soc - current_soc) * battery_capacity
        
        # Account for charging curve (slower at high SOC)
        if target_soc > 0.8:
            avg_power = charging_power * 0.7  # Reduced power at high SOC
        else:
            avg_power = charging_power * 0.9  # Account for inefficiencies
        
        charging_time = energy_needed / avg_power
        return max(0, charging_time)
    
    @staticmethod
    def calculate_cost_analysis(distance: float, efficiency: float,
                              electricity_price: float = 0.15,
                              gasoline_price: float = 1.5,
                              ice_efficiency: float = 8.0) -> Dict:
        """Calculate cost comparison between EV and ICE vehicle"""
        
        # EV costs
        energy_consumed = distance / efficiency if efficiency > 0 else 0
        ev_cost = energy_consumed * electricity_price
        
        # ICE costs (efficiency in km/L)
        fuel_consumed = distance / ice_efficiency if ice_efficiency > 0 else 0
        ice_cost = fuel_consumed * gasoline_price
        
        savings = ice_cost - ev_cost
        savings_percent = (savings / ice_cost) * 100 if ice_cost > 0 else 0
        
        return {
            'ev_cost': ev_cost,
            'ice_cost': ice_cost,
            'savings': savings,
            'savings_percent': savings_percent,
            'energy_consumed_kwh': energy_consumed,
            'fuel_consumed_liters': fuel_consumed
        }
    
    @staticmethod
    def generate_recommendations(battery_health: float, range_efficiency: float,
                               usage_pattern: str) -> List[str]:
        """Generate personalized recommendations based on vehicle data"""
        
        recommendations = []
        
        # Battery health recommendations
        if battery_health < 0.8:
            recommendations.append("🔋 Consider scheduling a battery health check")
            recommendations.append("⚡ Optimize charging patterns to extend battery life")
        
        if battery_health < 0.7:
            recommendations.append("🚨 Battery replacement may be needed soon")
            recommendations.append("♻️ Explore second-life battery applications")
        
        # Efficiency recommendations
        if range_efficiency < 3.5:  # km/kWh
            recommendations.append("🚗 Consider eco-driving techniques to improve efficiency")
            recommendations.append("🌡️ Pre-condition battery in extreme temperatures")
        
        # Usage pattern recommendations
        if usage_pattern == "aggressive":
            recommendations.append("🐌 Moderate acceleration can improve range by 15-20%")
            recommendations.append("⚡ Use regenerative braking effectively")
        
        if usage_pattern == "highway_heavy":
            recommendations.append("🛣️ Maintain steady speeds on highways for optimal efficiency")
            recommendations.append("🔄 Plan charging stops for long trips")
        
        # General recommendations
        recommendations.extend([
            "📱 Use mobile app for remote battery conditioning",
            "🔌 Charge to 80% for daily use, 100% for long trips",
            "❄️ Store vehicle in moderate temperatures when possible"
        ])
        
        return recommendations[:6]  # Return top 6 recommendations
    
    @staticmethod
    def format_time_duration(hours: float) -> str:
        """Format time duration in human-readable format"""
        
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes} minutes"
        elif hours < 24:
            hours_int = int(hours)
            minutes = int((hours - hours_int) * 60)
            if minutes > 0:
                return f"{hours_int}h {minutes}m"
            else:
                return f"{hours_int} hours"
        else:
            days = int(hours / 24)
            remaining_hours = int(hours % 24)
            if remaining_hours > 0:
                return f"{days} days {remaining_hours}h"
            else:
                return f"{days} days"
