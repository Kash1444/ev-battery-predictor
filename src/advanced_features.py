"""
Advanced Features for TATA Hackathon Demo
Additional impressive capabilities to showcase
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

class AdvancedFeatures:
    """Advanced features to impress hackathon judges"""
    
    @staticmethod
    def create_fleet_dashboard():
        """Create a fleet management dashboard"""
        st.subheader("🚛 Fleet Management Dashboard")
        
        # Generate mock fleet data
        fleet_data = {
            'vehicle_id': [f'TATA-EV-{i:03d}' for i in range(1, 21)],
            'battery_health': np.random.uniform(0.75, 0.98, 20),
            'current_range': np.random.uniform(150, 400, 20),
            'location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune'], 20),
            'status': np.random.choice(['Active', 'Charging', 'Maintenance'], 20, p=[0.7, 0.2, 0.1])
        }
        
        fleet_df = pd.DataFrame(fleet_data)
        
        # Fleet metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vehicles", len(fleet_df))
        with col2:
            avg_health = fleet_df['battery_health'].mean()
            st.metric("Avg Battery Health", f"{avg_health:.1%}")
        with col3:
            active_vehicles = len(fleet_df[fleet_df['status'] == 'Active'])
            st.metric("Active Vehicles", active_vehicles)
        with col4:
            maintenance_needed = len(fleet_df[fleet_df['battery_health'] < 0.8])
            st.metric("Maintenance Alerts", maintenance_needed)
        
        # Fleet map visualization (simulated)
        st.subheader("📍 Real-time Fleet Location")
        map_data = pd.DataFrame({
            'lat': [19.0760, 28.7041, 12.9716, 13.0827, 18.5204],
            'lon': [72.8777, 77.1025, 77.5946, 80.2707, 73.8567],
            'city': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune'],
            'vehicles': [4, 5, 3, 4, 4]
        })
        
        st.map(map_data[['lat', 'lon']])
        
        # Fleet table
        st.subheader("🚗 Vehicle Status")
        st.dataframe(fleet_df.style.format({
            'battery_health': '{:.1%}',
            'current_range': '{:.0f} km'
        }))
        
        return fleet_df
    
    @staticmethod
    def create_sustainability_tracker():
        """Create sustainability impact tracker"""
        st.subheader("🌱 Sustainability Impact Tracker")
        
        # Calculate environmental benefits
        total_km = st.number_input("Fleet Total Distance (km/month)", 10000, 100000, 50000)
        
        # Environmental calculations
        ev_energy = total_km * 0.2  # kWh per km
        ice_fuel = total_km * 0.08  # liters per km
        
        co2_ev = ev_energy * 0.5  # kg CO2 (assuming grid mix)
        co2_ice = ice_fuel * 2.3  # kg CO2 per liter
        
        co2_saved = co2_ice - co2_ev
        trees_equivalent = co2_saved / 21  # kg CO2 absorbed per tree per year
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CO2 Saved", f"{co2_saved:,.0f} kg", delta="vs ICE fleet")
        with col2:
            st.metric("Trees Equivalent", f"{trees_equivalent:,.0f} trees/year")
        with col3:
            cost_savings = ice_fuel * 1.5 - ev_energy * 0.12  # Fuel vs electricity cost
            st.metric("Cost Savings", f"₹{cost_savings:,.0f}/month")
        
        # Sustainability timeline
        months = list(range(1, 13))
        cumulative_co2 = [co2_saved * m for m in months]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_co2,
            mode='lines+markers',
            name='Cumulative CO2 Savings',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="Annual CO2 Savings Impact",
            xaxis_title="Month",
            yaxis_title="CO2 Saved (kg)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return {
            'co2_saved': co2_saved,
            'cost_savings': cost_savings,
            'trees_equivalent': trees_equivalent
        }
    
    @staticmethod
    def create_predictive_maintenance():
        """Create predictive maintenance alerts"""
        st.subheader("🔧 Predictive Maintenance AI")
        
        # Simulate maintenance predictions
        components = ['Battery Pack', 'Motor', 'Inverter', 'Cooling System', 'Charging Port']
        
        maintenance_data = []
        for component in components:
            health_score = np.random.uniform(0.6, 0.98)
            days_to_maintenance = int(np.random.exponential(100))
            
            if health_score < 0.8:
                urgency = "High"
                color = "🔴"
            elif health_score < 0.9:
                urgency = "Medium"
                color = "🟡"
            else:
                urgency = "Low"
                color = "🟢"
            
            maintenance_data.append({
                'Component': component,
                'Health Score': health_score,
                'Days to Service': days_to_maintenance,
                'Urgency': urgency,
                'Status': color
            })
        
        maintenance_df = pd.DataFrame(maintenance_data)
        
        # Display maintenance alerts
        for _, row in maintenance_df.iterrows():
            if row['Urgency'] == 'High':
                st.error(f"{row['Status']} {row['Component']}: Service needed in {row['Days to Service']} days")
            elif row['Urgency'] == 'Medium':
                st.warning(f"{row['Status']} {row['Component']}: Service due in {row['Days to Service']} days")
            else:
                st.success(f"{row['Status']} {row['Component']}: Good condition")
        
        # Maintenance schedule visualization
        fig = go.Figure()
        
        for i, row in maintenance_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Component']],
                y=[row['Health Score']],
                name=row['Component'],
                marker_color='red' if row['Urgency'] == 'High' else 'orange' if row['Urgency'] == 'Medium' else 'green'
            ))
        
        fig.update_layout(
            title="Component Health Status",
            yaxis_title="Health Score",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return maintenance_df
    
    @staticmethod
    def create_ai_chatbot():
        """Create an AI chatbot for EV assistance"""
        st.subheader("🤖 TATA EV Assistant")
        
        # Sample Q&A responses
        responses = {
            "range": "Based on current conditions, your vehicle can travel approximately 285 km. Would you like route optimization suggestions?",
            "charging": "I recommend charging to 80% for daily use. The optimal charging time is 45 minutes using fast charging.",
            "battery": "Your battery health is excellent at 94%. Expected lifespan: 8+ years with current usage patterns.",
            "maintenance": "Next service due in 67 days. No immediate issues detected. Tire pressure check recommended.",
            "efficiency": "To improve efficiency: Use eco mode, maintain steady speeds, and pre-condition while plugged in.",
            "weather": "Cold weather detected. Range may be reduced by 15%. Consider preheating the cabin while charging.",
            "trip": "For your 150km trip, departure with 85% charge is recommended. Charging station backup identified at 120km mark."
        }
        
        # Chat interface
        user_input = st.text_input("Ask your TATA EV Assistant:", placeholder="e.g., How's my battery health?")
        
        if user_input:
            # Simple keyword matching for demo
            response = "I'm here to help with your EV questions! Try asking about range, charging, battery, maintenance, efficiency, weather, or trip planning."
            
            for keyword, reply in responses.items():
                if keyword.lower() in user_input.lower():
                    response = reply
                    break
            
            st.chat_message("assistant").write(response)
            
            # Show suggested questions
            st.subheader("💡 Suggested Questions:")
            suggestions = [
                "What's my current range?",
                "When should I charge my battery?",
                "How's my battery health?",
                "When is my next maintenance?",
                "How can I improve efficiency?",
                "How does weather affect my range?",
                "Plan my trip to Delhi"
            ]
            
            for suggestion in suggestions:
                if st.button(suggestion, key=f"btn_{suggestion}"):
                    for keyword, reply in responses.items():
                        if keyword.lower() in suggestion.lower():
                            st.chat_message("assistant").write(reply)
                            break
    
    @staticmethod
    def create_carbon_credits():
        """Calculate and display carbon credits earned"""
        st.subheader("💰 Carbon Credits Calculator")
        
        # Input parameters
        monthly_km = st.slider("Monthly Distance (km)", 500, 5000, 1500)
        
        # Calculate carbon savings
        ev_emissions = monthly_km * 0.1  # kg CO2 per km for EV
        ice_emissions = monthly_km * 0.184  # kg CO2 per km for ICE
        
        carbon_saved = ice_emissions - ev_emissions
        carbon_credits = carbon_saved * 0.001  # Convert to tons
        credit_value = carbon_credits * 50  # USD per ton
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Carbon Saved", f"{carbon_saved:.1f} kg CO2/month")
        with col2:
            st.metric("Carbon Credits", f"{carbon_credits:.3f} tons/month")
        with col3:
            st.metric("Credit Value", f"${credit_value:.2f}/month")
        
        # Annual projection
        annual_credits = carbon_credits * 12
        annual_value = credit_value * 12
        
        st.success(f"🌱 Annual Projection: {annual_credits:.2f} tons CO2 credits worth ${annual_value:.0f}")
        
        # Carbon credits visualization
        months = list(range(1, 13))
        cumulative_credits = [carbon_credits * m for m in months]
        cumulative_value = [credit_value * m for m in months]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_credits,
            mode='lines+markers',
            name='Carbon Credits (tons)',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_value,
            mode='lines+markers',
            name='Credit Value ($)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Annual Carbon Credits Accumulation",
            xaxis_title="Month",
            yaxis=dict(title="Carbon Credits (tons)", side="left"),
            yaxis2=dict(title="Value ($)", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return {
            'monthly_credits': carbon_credits,
            'monthly_value': credit_value,
            'annual_credits': annual_credits,
            'annual_value': annual_value
        }

def add_advanced_features_tab():
    """Add advanced features to the main app"""
    st.subheader("🚀 Advanced AI Features")
    
    # Feature selector
    feature_options = [
        "🚛 Fleet Management",
        "🌱 Sustainability Tracker", 
        "🔧 Predictive Maintenance",
        "🤖 AI Assistant",
        "💰 Carbon Credits"
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
