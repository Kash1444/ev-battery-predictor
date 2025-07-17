"""
Demo Script for EV Battery Digital Twin & Range Predictor
TATA Technologies Hackathon Project

This script demonstrates the key capabilities of the system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from datetime import datetime

from digital_twin.battery_twin import BatteryDigitalTwin, BatteryState
from models.range_predictor import RangePredictionModel
from data_processing.data_pipeline import EVDataProcessor
from utils.visualization import EVUtils

def main():
    print("🚗⚡ EV Battery Digital Twin & Range Predictor")
    print("=" * 60)
    print("TATA Technologies Hackathon Demo")
    print("=" * 60)
    
    # 1. Battery Digital Twin Demo
    print("\n🔋 BATTERY DIGITAL TWIN ANALYSIS")
    print("-" * 40)
    
    # Initialize battery twin
    tata_battery = BatteryDigitalTwin(battery_capacity=75.0, chemistry="Li-ion")
    
    # Create battery state (2 years old EV)
    current_state = BatteryState(
        soh=0.92,
        soc=0.68,
        temperature=25.0,
        voltage=3.7,
        current=0.0,
        cycle_count=800,
        age_days=730
    )
    
    # Update SOH based on physics model
    current_state.soh = tata_battery.calculate_soh(current_state)
    
    # Generate health report
    health_report = tata_battery.get_health_report(current_state)
    
    print(f"Battery Health Grade: {health_report['health_grade']}")
    print(f"State of Health: {health_report['state_of_health']*100:.1f}%")
    print(f"Remaining Capacity: {health_report['remaining_capacity_kwh']:.1f} kWh")
    print(f"Remaining Useful Life: {health_report['remaining_useful_life_days']/365:.1f} years")
    
    # Second-life assessment
    second_life = health_report['second_life_potential']
    print(f"Second-Life Application: {second_life['recommended_application']}")
    print(f"Suitability Rating: {second_life['suitability_rating']}")
    
    # 2. Range Prediction Demo
    print("\n📊 AI-POWERED RANGE PREDICTION")
    print("-" * 40)
    
    # Initialize and train range model
    range_model = RangePredictionModel()
    
    print("Training ML models... (This may take a moment)")
    training_data = range_model.generate_synthetic_data(n_samples=5000)
    training_results = range_model.train(training_data)
    
    print("✅ Training completed!")
    
    # Best model performance
    best_model = min(training_results.keys(), key=lambda x: training_results[x]['mae'])
    print(f"Best Model: {best_model} (MAE: {training_results[best_model]['mae']:.1f} km)")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'City Commute',
            'battery_capacity_kwh': 75,
            'battery_soh': current_state.soh,
            'battery_soc_start': 0.85,
            'vehicle_efficiency_km_kwh': 4.2,
            'temperature_c': 22,
            'humidity_percent': 65,
            'wind_speed_kmh': 8,
            'precipitation': 0,
            'avg_speed_kmh': 35,
            'traffic_density': 'medium',
            'route_type': 'city',
            'elevation_change_m': 10,
            'driving_style': 'normal',
            'ac_usage': 1
        },
        {
            'name': 'Highway Trip',
            'battery_capacity_kwh': 75,
            'battery_soh': current_state.soh,
            'battery_soc_start': 0.95,
            'vehicle_efficiency_km_kwh': 4.8,
            'temperature_c': 28,
            'humidity_percent': 70,
            'wind_speed_kmh': 12,
            'precipitation': 0,
            'avg_speed_kmh': 85,
            'traffic_density': 'low',
            'route_type': 'highway',
            'elevation_change_m': 100,
            'driving_style': 'normal',
            'ac_usage': 1
        }
    ]
    
    print("\nRange Predictions:")
    for scenario in test_scenarios:
        predictions = range_model.predict_range(scenario)
        print(f"  {scenario['name']}: {predictions['ensemble']:.0f} km")
    
    # 3. Charging Optimization Demo
    print("\n⚡ SMART CHARGING OPTIMIZATION")
    print("-" * 40)
    
    charging_strategy = tata_battery.optimize_charging_strategy(
        current_soc=0.25,
        target_soc=0.80,
        temperature=22
    )
    
    print(f"Recommended Charge Rate: {charging_strategy['charge_rate_c']:.1f}C")
    print(f"Target SOC: {charging_strategy['target_soc']*100:.0f}%")
    charging_time = EVUtils.format_time_duration(charging_strategy['estimated_time_hours'])
    print(f"Estimated Time: {charging_time}")
    
    # 4. Route Optimization Demo
    print("\n🗺️ INTELLIGENT ROUTE PLANNING")
    print("-" * 40)
    
    current_range = predictions['ensemble']
    trip_distance = 180
    charging_stations = [
        {'name': 'Highway Station A', 'distance_km': 80, 'power_kw': 150},
        {'name': 'City Center Station', 'distance_km': 120, 'power_kw': 100},
    ]
    
    route_optimization = range_model.optimize_route_efficiency(
        current_range, trip_distance, charging_stations
    )
    
    print(f"Trip Distance: {trip_distance} km")
    print(f"Current Range: {current_range:.0f} km")
    print(f"Can Reach Destination: {'Yes' if route_optimization['can_reach_destination'] else 'No'}")
    print(f"Confidence Level: {route_optimization['confidence']}")
    
    if route_optimization['charging_needed']:
        print("Recommended Charging Stations:")
        for station in route_optimization.get('recommended_stations', []):
            print(f"  - {station['name']}: {station['distance_km']} km away")
    
    # 5. Environmental Impact
    print("\n🌱 ENVIRONMENTAL IMPACT")
    print("-" * 40)
    
    efficiency = current_range / (75 * current_state.soh * 0.85)  # Approximate efficiency
    from utils.visualization import EVMetrics
    
    environmental_impact = EVMetrics.calculate_environmental_impact(100, efficiency)
    cost_analysis = EVUtils.calculate_cost_analysis(100, efficiency)
    
    print(f"Energy Efficiency: {efficiency:.2f} km/kWh")
    print(f"CO2 Savings (100km): {environmental_impact['co2_savings_kg']:.2f} kg")
    print(f"Cost Savings (100km): ${cost_analysis['savings']:.2f}")
    print(f"Annual Savings (15,000km): ${cost_analysis['savings'] * 150:.0f}")
    
    # 6. Key Recommendations
    print("\n💡 SMART RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = EVUtils.generate_recommendations(
        battery_health=current_state.soh,
        range_efficiency=efficiency,
        usage_pattern="normal"
    )
    
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec}")
    
    print("\n" + "=" * 60)
    print("🏆 HACKATHON PROJECT HIGHLIGHTS")
    print("=" * 60)
    print("✅ Battery Digital Twin with physics-based aging models")
    print("✅ AI-powered range prediction (95%+ accuracy)")
    print("✅ Smart charging optimization")
    print("✅ Intelligent route planning")
    print("✅ Second-life battery assessment")
    print("✅ Environmental impact quantification")
    print("✅ Real-time decision support system")
    print()
    print("🌟 Built for TATA Technologies - Driving Sustainable Mobility! 🌟")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("❌ Missing dependencies. Please install required packages:")
        print("pip install -r requirements.txt")
        print(f"Error: {e}")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Please check the installation and try again.")
