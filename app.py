"""
EV Battery Digital Twin & Range Predictor
TATA Technologies Hackathon Project

Advanced ML application for EV battery health monitoring and range optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from digital_twin.battery_twin import BatteryDigitalTwin, BatteryState
from models.range_predictor import RangePredictionModel
from data_processing.data_pipeline import EVDataProcessor
from utils.visualization import EVVisualization, EVMetrics, EVUtils
from advanced_features import AdvancedFeatures

# Configure page
st.set_page_config(
    page_title="EV Battery Digital Twin",
    page_icon="🚗⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 1rem;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_demo_data():
    """Load demo data for the application"""
    processor = EVDataProcessor()
    data = processor.generate_realistic_ev_dataset(n_vehicles=50, days_per_vehicle=180)
    data = processor.clean_and_validate_data(data)
    data = processor.engineer_features(data)
    return data, processor

@st.cache_resource
def train_models():
    """Train ML models for the application"""
    # Load and prepare data
    data, processor = load_demo_data()
    
    # Train range prediction model
    range_model = RangePredictionModel()
    model_data = range_model.generate_synthetic_data(n_samples=5000)
    training_results = range_model.train(model_data)
    
    return range_model, processor, training_results

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚗⚡ EV Battery Digital Twin & Range Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h3>TATA Technologies Hackathon Project</h3>
        <p>Advanced ML solution for sustainable mobility and circular battery systems</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize models
    with st.spinner("🔄 Loading AI models and data..."):
        range_model, processor, training_results = train_models()
        battery_twin = BatteryDigitalTwin()
    
    # Sidebar for inputs
    st.sidebar.markdown("## 🔧 Vehicle Configuration")
    
    # Vehicle parameters
    with st.sidebar.expander("🚗 Vehicle Settings", expanded=True):
        vehicle_type = st.selectbox("Vehicle Type", ["compact", "sedan", "suv"])
        battery_capacity = st.slider("Battery Capacity (kWh)", 40, 120, 75)
        vehicle_age_months = st.slider("Vehicle Age (months)", 0, 60, 12)
        
    # Battery state
    with st.sidebar.expander("🔋 Battery State", expanded=True):
        current_soc = st.slider("Current Charge Level (%)", 10, 100, 65) / 100
        cycle_count = st.number_input("Total Charge Cycles", 0, 2000, 300)
        
    # Environmental conditions
    with st.sidebar.expander("🌤️ Environmental Conditions", expanded=True):
        temperature = st.slider("Temperature (°C)", -20, 45, 22)
        humidity = st.slider("Humidity (%)", 20, 95, 60)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
        precipitation = st.selectbox("Precipitation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
    # Trip parameters
    with st.sidebar.expander("🛣️ Trip Planning", expanded=True):
        trip_distance = st.slider("Planned Distance (km)", 10, 300, 80)
        route_type = st.selectbox("Route Type", ["city", "highway", "mixed"])
        traffic_density = st.selectbox("Traffic Density", ["low", "medium", "high"])
        driving_style = st.selectbox("Driving Style", ["eco", "normal", "aggressive"])
        ac_usage = st.checkbox("Air Conditioning On", value=True)
        
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔋 Battery Health", "📊 Range Prediction", "⚡ Charging Optimization", "📈 Analytics", "🚀 Advanced AI"])
        
        # Battery Health Tab
        with tab1:
            st.subheader("🔋 Battery Digital Twin Analysis")
            
            # Create battery state
            battery_state = BatteryState(
                soh=0.85,  # Will be calculated
                soc=current_soc,
                temperature=temperature,
                voltage=3.7,
                current=0.0,
                cycle_count=cycle_count,
                age_days=vehicle_age_months * 30
            )
            
            # Get battery health report
            health_report = battery_twin.get_health_report(battery_state)
            
            # Display metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Health Grade", health_report['health_grade'], 
                         help="Overall battery health grade")
            
            with col_b:
                soh_percent = health_report['state_of_health'] * 100
                st.metric("State of Health", f"{soh_percent:.1f}%",
                         delta=f"{soh_percent - 90:.1f}%")
            
            with col_c:
                st.metric("Remaining Capacity", 
                         f"{health_report['remaining_capacity_kwh']:.1f} kWh",
                         help="Usable battery capacity")
            
            with col_d:
                rul_years = health_report['remaining_useful_life_days'] / 365
                st.metric("Remaining Life", f"{rul_years:.1f} years",
                         help="Estimated remaining useful life")
            
            # Battery simulation
            st.subheader("📊 Battery Life Simulation")
            simulation_years = st.slider("Simulation Period (years)", 1, 15, 10)
            daily_cycles = st.slider("Daily Cycles", 0.5, 3.0, 1.2, 0.1)
            avg_temp = st.slider("Average Temperature (°C)", 5, 40, 25)
            
            simulation_data = battery_twin.simulate_battery_life(
                years=simulation_years, 
                daily_cycles=daily_cycles,
                avg_temperature=avg_temp
            )
            
            # Visualization
            fig_health = EVVisualization.plot_battery_health_timeline(simulation_data)
            st.plotly_chart(fig_health, use_container_width=True)
            
            # Second-life analysis
            st.subheader("♻️ Second-Life Battery Assessment")
            second_life = health_report['second_life_potential']
            
            col_sl1, col_sl2 = st.columns(2)
            with col_sl1:
                st.info(f"**Recommended Application:** {second_life['recommended_application']}")
                st.info(f"**Suitability Rating:** {second_life['suitability_rating']}")
            
            with col_sl2:
                st.success(f"**Remaining Capacity:** {second_life['remaining_capacity_kwh']:.1f} kWh")
                st.success(f"**Estimated Second-Life:** {second_life['estimated_second_life_years']} years")
        
        # Range Prediction Tab
        with tab2:
            st.subheader("📊 AI-Powered Range Prediction")
            
            # Prepare input data
            input_data = {
                'battery_capacity_kwh': battery_capacity,
                'battery_soh': health_report['state_of_health'],
                'battery_soc_start': current_soc,
                'vehicle_efficiency_km_kwh': 4.5,  # Default efficiency
                'temperature_c': temperature,
                'humidity_percent': humidity,
                'wind_speed_kmh': wind_speed,
                'precipitation': precipitation,
                'avg_speed_kmh': 60,  # Default speed
                'traffic_density': traffic_density,
                'route_type': route_type,
                'elevation_change_m': 0,  # Default
                'driving_style': driving_style,
                'ac_usage': int(ac_usage)
            }
            
            # Get predictions
            predictions = range_model.predict_range(input_data)
            
            # Display predictions
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                st.metric("🎯 AI Prediction", f"{predictions['ensemble']:.0f} km",
                         help="Ensemble model prediction")
            
            with col_r2:
                st.metric("🌲 Random Forest", f"{predictions['random_forest']:.0f} km",
                         help="Random Forest model")
            
            with col_r3:
                st.metric("🚀 Gradient Boost", f"{predictions['gradient_boost']:.0f} km",
                         help="Gradient Boosting model")
            
            with col_r4:
                confidence = "High" if abs(predictions['random_forest'] - predictions['gradient_boost']) < 20 else "Medium"
                st.metric("🎯 Confidence", confidence,
                         help="Prediction confidence level")
            
            # Range visualization
            st.subheader("📈 Range Analysis")
            
            # Create gauge chart for range
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = predictions['ensemble'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Predicted Range (km)"},
                delta = {'reference': 300},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 150], 'color': "lightgray"},
                        {'range': [150, 300], 'color': "yellow"},
                        {'range': [300, 500], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 400
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Factors analysis
            st.subheader("🔍 Range Factors Analysis")
            factors_analysis = range_model.get_range_factors_analysis(input_data)
            
            if factors_analysis:
                # Display top factors
                factors_df = pd.DataFrame(factors_analysis['top_factors'])
                
                fig_factors = px.bar(
                    factors_df, 
                    x='importance', 
                    y='factor',
                    orientation='h',
                    title="Top Factors Affecting Range",
                    color='importance',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_factors, use_container_width=True)
            
            # Route optimization
            st.subheader("🗺️ Route Optimization")
            
            # Sample charging stations
            charging_stations = [
                {'name': 'Station A', 'distance_km': 45, 'power_kw': 150},
                {'name': 'Station B', 'distance_km': 78, 'power_kw': 50},
                {'name': 'Station C', 'distance_km': 95, 'power_kw': 100}
            ]
            
            route_optimization = range_model.optimize_route_efficiency(
                predictions['ensemble'], trip_distance, charging_stations
            )
            
            if route_optimization['can_reach_destination']:
                st.success(f"✅ Destination reachable with {route_optimization['confidence'].lower()} confidence")
                if route_optimization['charging_needed']:
                    st.warning("⚡ Charging recommended en route")
                    for station in route_optimization['recommended_stations']:
                        st.info(f"🔌 {station['name']} - {station['distance_km']} km away")
            else:
                st.error("❌ Destination may not be reachable - charging required")
        
        # Charging Optimization Tab
        with tab3:
            st.subheader("⚡ Smart Charging Optimization")
            
            target_soc = st.slider("Target Charge Level (%)", 50, 100, 80) / 100
            
            charging_strategy = battery_twin.optimize_charging_strategy(
                current_soc, target_soc, temperature
            )
            
            # Display charging recommendations
            col_c1, col_c2, col_c3 = st.columns(3)
            
            with col_c1:
                st.metric("Charge Rate", f"{charging_strategy['charge_rate_c']:.1f}C",
                         help="Recommended charging rate")
            
            with col_c2:
                st.metric("Max Voltage", f"{charging_strategy['max_voltage']:.2f}V",
                         help="Maximum charging voltage")
            
            with col_c3:
                charging_time = charging_strategy['estimated_time_hours']
                time_formatted = EVUtils.format_time_duration(charging_time)
                st.metric("Estimated Time", time_formatted,
                         help="Estimated charging time")
            
            # Charging curve visualization
            fig_charging = EVVisualization.plot_charging_optimization(charging_strategy)
            st.plotly_chart(fig_charging, use_container_width=True)
            
            # Charging recommendations
            st.subheader("💡 Smart Charging Tips")
            
            if charging_strategy['temperature_compensation']:
                st.warning("🌡️ Temperature compensation active - slower charging for battery protection")
            
            if target_soc > 0.9:
                st.info(f"💡 Recommended target: {charging_strategy['target_soc']*100:.0f}% for battery longevity")
            
            charging_tips = [
                "🔋 Charge to 80% for daily use, 100% only for long trips",
                "❄️ Pre-condition battery in cold weather",
                "🌡️ Avoid charging in extreme temperatures when possible",
                "⚡ Use AC charging overnight for battery health",
                "📱 Monitor charging progress via mobile app"
            ]
            
            for tip in charging_tips:
                st.info(tip)
        
        # Analytics Tab
        with tab4:
            st.subheader("📈 Advanced Analytics")
            
            # Model performance
            if training_results:
                st.subheader("🤖 Model Performance Metrics")
                
                metrics_df = pd.DataFrame(training_results).T
                st.dataframe(metrics_df.round(3))
                
                # Best model
                best_model = min(training_results.keys(), 
                               key=lambda x: training_results[x]['mae'])
                st.success(f"🏆 Best performing model: {best_model.title()}")
            
            # Environmental impact
            st.subheader("🌱 Environmental Impact")
            
            efficiency = predictions['ensemble'] / (battery_capacity * health_report['state_of_health'])
            environmental_impact = EVMetrics.calculate_environmental_impact(
                trip_distance, efficiency
            )
            
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                st.metric("Energy Consumption", 
                         f"{environmental_impact['energy_consumed_kwh']:.1f} kWh")
                st.metric("CO2 Emissions", 
                         f"{environmental_impact['co2_emissions_kg']:.1f} kg")
            
            with col_e2:
                st.metric("CO2 Savings vs ICE", 
                         f"{environmental_impact['co2_savings_kg']:.1f} kg")
                st.metric("Emissions Reduction", 
                         f"{environmental_impact['emissions_reduction_percent']:.1f}%")
            
            # Cost analysis
            st.subheader("💰 Cost Analysis")
            
            cost_analysis = EVUtils.calculate_cost_analysis(trip_distance, efficiency)
            
            col_cost1, col_cost2 = st.columns(2)
            
            with col_cost1:
                st.metric("EV Trip Cost", f"${cost_analysis['ev_cost']:.2f}")
                st.metric("ICE Trip Cost", f"${cost_analysis['ice_cost']:.2f}")
            
            with col_cost2:
                st.metric("Cost Savings", f"${cost_analysis['savings']:.2f}")
                st.metric("Savings Percentage", f"{cost_analysis['savings_percent']:.1f}%")
        
        # Advanced AI Features Tab
        with tab5:
            st.subheader("🚀 Advanced AI Features")
            
            # Feature selector
            feature_options = [
                "🚛 Fleet Management Dashboard",
                "🌱 Sustainability Impact Tracker", 
                "🔧 Predictive Maintenance AI",
                "🤖 TATA EV Assistant",
                "💰 Carbon Credits Calculator"
            ]
            
            selected_feature = st.selectbox("Choose Advanced Feature:", feature_options)
            
            features = AdvancedFeatures()
            
            if "Fleet Management" in selected_feature:
                features.create_fleet_dashboard()
            elif "Sustainability" in selected_feature:
                features.create_sustainability_tracker()
            elif "Maintenance" in selected_feature:
                features.create_predictive_maintenance()
            elif "Assistant" in selected_feature:
                features.create_ai_chatbot()
            elif "Carbon Credits" in selected_feature:
                features.create_carbon_credits()
    
    # Right sidebar with recommendations
    with col2:
        st.markdown("## 💡 Smart Recommendations")
        
        # Generate recommendations
        recommendations = EVUtils.generate_recommendations(
            health_report['state_of_health'],
            efficiency,
            driving_style
        )
        
        for i, rec in enumerate(recommendations):
            st.markdown(f'<div class="recommendation-box">{rec}</div>', 
                       unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("## 📊 Quick Stats")
        
        stats_data = {
            "Battery Health": f"{health_report['state_of_health']*100:.1f}%",
            "Predicted Range": f"{predictions['ensemble']:.0f} km",
            "Efficiency": f"{efficiency:.1f} km/kWh",
            "Temperature": f"{temperature}°C",
            "Trip Distance": f"{trip_distance} km"
        }
        
        for stat, value in stats_data.items():
            st.metric(stat, value)
        
        # Emergency contacts (demo)
        st.markdown("## 🆘 Emergency Support")
        st.info("🔧 TATA Service: 1-800-TATA")
        st.info("⚡ Charging Support: 1-800-CHARGE")
        st.info("🗺️ Route Assistance: 1-800-ROUTE")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <strong>EV Battery Digital Twin & Range Predictor</strong><br>
        Built for TATA Technologies Hackathon | Powered by Advanced ML & AI<br>
        🌱 Driving Sustainable Mobility & Circular Battery Systems
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()