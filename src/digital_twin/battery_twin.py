"""
Battery Digital Twin Engine
Advanced modeling for battery aging, health prediction, and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BatteryState:
    """Battery state representation"""
    soh: float  # State of Health (0-1)
    soc: float  # State of Charge (0-1)
    temperature: float  # Temperature in Celsius
    voltage: float  # Voltage
    current: float  # Current
    cycle_count: int  # Number of charge cycles
    age_days: int  # Age in days

class BatteryDigitalTwin:
    """
    Digital Twin for EV Battery Modeling and Prediction
    
    This class models battery degradation, predicts remaining useful life,
    and optimizes charging strategies using real-world data patterns.
    """
    
    def __init__(self, battery_capacity: float = 75.0, chemistry: str = "Li-ion"):
        self.battery_capacity = battery_capacity  # kWh
        self.chemistry = chemistry
        self.initial_capacity = battery_capacity
        
        # Battery degradation parameters (based on research data)
        self.calendar_aging_factor = 2.5e-5  # per day
        self.cycle_aging_factor = 8.5e-6     # per cycle
        self.temperature_factor = 0.08       # temperature impact
        self.depth_of_discharge_factor = 0.15 # DoD impact
        
        self.state_history = []
        self.predictions = []
        
    def calculate_soh(self, state: BatteryState) -> float:
        """
        Calculate State of Health based on aging factors
        
        SOH degradation model incorporating:
        - Calendar aging (time-based)
        - Cycle aging (usage-based)  
        - Temperature stress
        - Depth of discharge impact
        """
        
        # Calendar aging component
        calendar_degradation = self.calendar_aging_factor * state.age_days
        
        # Cycle aging component
        cycle_degradation = self.cycle_aging_factor * state.cycle_count
        
        # Temperature stress (Arrhenius model)
        temp_stress = np.exp(self.temperature_factor * (state.temperature - 25) / 25)
        
        # DoD stress (assuming average 80% DoD)
        dod_stress = 1 + self.depth_of_discharge_factor * 0.8
        
        # Combined degradation
        total_degradation = (calendar_degradation + cycle_degradation) * temp_stress * dod_stress
        
        # SOH = 1 - total_degradation (with minimum threshold)
        soh = max(0.7, 1.0 - total_degradation)
        
        return soh
    
    def predict_remaining_useful_life(self, current_state: BatteryState, 
                                    end_of_life_threshold: float = 0.8) -> int:
        """
        Predict Remaining Useful Life (RUL) in days
        
        Args:
            current_state: Current battery state
            end_of_life_threshold: SOH threshold for end of life
            
        Returns:
            Remaining useful life in days
        """
        
        current_soh = self.calculate_soh(current_state)
        
        if current_soh <= end_of_life_threshold:
            return 0
        
        # Estimate daily degradation rate
        daily_degradation = self.calendar_aging_factor
        cycle_degradation_per_day = self.cycle_aging_factor * 1  # Assume 1 cycle per day
        
        total_daily_degradation = daily_degradation + cycle_degradation_per_day
        
        # Calculate days to reach threshold
        soh_to_lose = current_soh - end_of_life_threshold
        rul_days = int(soh_to_lose / total_daily_degradation)
        
        return max(0, rul_days)
    
    def optimize_charging_strategy(self, current_soc: float, 
                                 target_soc: float = 0.8,
                                 temperature: float = 25.0) -> Dict:
        """
        Optimize charging strategy for battery longevity
        
        Args:
            current_soc: Current state of charge
            target_soc: Target state of charge
            temperature: Ambient temperature
            
        Returns:
            Optimized charging parameters
        """
        
        # Optimal charging parameters based on battery research
        if temperature < 0:
            charge_rate = 0.1  # Slow charging in cold
            max_voltage = 4.1
        elif temperature > 35:
            charge_rate = 0.3  # Reduced rate in heat
            max_voltage = 4.05
        else:
            charge_rate = 0.5  # Normal charging
            max_voltage = 4.15
        
        # Adjust target SOC for longevity
        if target_soc > 0.9:
            recommended_target = 0.85  # Avoid high SOC stress
        else:
            recommended_target = target_soc
        
        charging_time = (recommended_target - current_soc) * self.battery_capacity / charge_rate
        
        return {
            'charge_rate_c': charge_rate,
            'max_voltage': max_voltage,
            'target_soc': recommended_target,
            'estimated_time_hours': charging_time,
            'temperature_compensation': temperature < 0 or temperature > 35
        }
    
    def assess_second_life_potential(self, state: BatteryState) -> Dict:
        """
        Assess battery potential for second-life applications
        
        Args:
            state: Current battery state
            
        Returns:
            Second-life assessment with recommendations
        """
        
        soh = self.calculate_soh(state)
        
        # Second-life application mapping
        if soh >= 0.8:
            application = "EV Use - Continue"
            suitability = "Excellent"
        elif soh >= 0.7:
            application = "Grid Storage"
            suitability = "Good"
        elif soh >= 0.6:
            application = "Home Energy Storage"
            suitability = "Moderate"
        elif soh >= 0.5:
            application = "UPS/Backup Power"
            suitability = "Limited"
        else:
            application = "Material Recovery"
            suitability = "End of Life"
        
        # Calculate remaining capacity
        remaining_capacity = soh * self.initial_capacity
        
        return {
            'current_soh': soh,
            'remaining_capacity_kwh': remaining_capacity,
            'recommended_application': application,
            'suitability_rating': suitability,
            'estimated_second_life_years': max(0, int((soh - 0.5) * 10)) if soh > 0.5 else 0
        }
    
    def simulate_battery_life(self, years: int = 10, 
                            daily_cycles: float = 1.0,
                            avg_temperature: float = 25.0) -> pd.DataFrame:
        """
        Simulate battery degradation over time
        
        Args:
            years: Simulation period in years
            daily_cycles: Average daily charge cycles
            avg_temperature: Average operating temperature
            
        Returns:
            DataFrame with simulation results
        """
        
        days = years * 365
        simulation_data = []
        
        for day in range(0, days, 30):  # Monthly snapshots
            state = BatteryState(
                soh=1.0,  # Will be calculated
                soc=0.5,  # Average SOC
                temperature=avg_temperature + np.random.normal(0, 5),  # Temperature variation
                voltage=3.7,
                current=0.0,
                cycle_count=int(day * daily_cycles),
                age_days=day
            )
            
            soh = self.calculate_soh(state)
            rul = self.predict_remaining_useful_life(state)
            capacity = soh * self.initial_capacity
            
            simulation_data.append({
                'day': day,
                'month': day // 30,
                'soh': soh,
                'capacity_kwh': capacity,
                'cycle_count': state.cycle_count,
                'rul_days': rul,
                'temperature': state.temperature
            })
        
        return pd.DataFrame(simulation_data)
    
    def get_health_report(self, state: BatteryState) -> Dict:
        """
        Generate comprehensive battery health report
        
        Args:
            state: Current battery state
            
        Returns:
            Comprehensive health report
        """
        
        soh = self.calculate_soh(state)
        rul = self.predict_remaining_useful_life(state)
        charging_strategy = self.optimize_charging_strategy(state.soc, temperature=state.temperature)
        second_life = self.assess_second_life_potential(state)
        
        # Health grade
        if soh >= 0.9:
            health_grade = "A"
        elif soh >= 0.8:
            health_grade = "B"
        elif soh >= 0.7:
            health_grade = "C"
        else:
            health_grade = "D"
        
        return {
            'health_grade': health_grade,
            'state_of_health': soh,
            'remaining_capacity_kwh': soh * self.initial_capacity,
            'remaining_useful_life_days': rul,
            'cycle_count': state.cycle_count,
            'age_days': state.age_days,
            'temperature_status': "Normal" if 0 <= state.temperature <= 40 else "Extreme",
            'charging_optimization': charging_strategy,
            'second_life_potential': second_life,
            'recommendations': self._generate_recommendations(soh, state)
        }
    
    def _generate_recommendations(self, soh: float, state: BatteryState) -> List[str]:
        """Generate maintenance and usage recommendations"""
        recommendations = []
        
        if soh < 0.8:
            recommendations.append("Consider battery replacement planning")
        
        if state.temperature > 35:
            recommendations.append("Monitor cooling system - high temperature detected")
        elif state.temperature < 0:
            recommendations.append("Avoid fast charging in cold conditions")
        
        if state.cycle_count > 1000:
            recommendations.append("High cycle count - monitor degradation closely")
        
        if state.soc > 0.9:
            recommendations.append("Avoid keeping battery at high SOC for extended periods")
        
        recommendations.append("Regular conditioning cycles recommended")
        recommendations.append("Maintain optimal operating temperature (15-25°C)")
        
        return recommendations
