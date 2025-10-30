"""
Synthetic Power Grid Data Generator
Generates realistic power grid data for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_power_grid_data(n_samples=5000, random_state=42):
    """
    Generate synthetic power grid data with realistic patterns
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Generated power grid dataset
    """
    np.random.seed(random_state)
    
    # Generate base features
    voltage = np.random.normal(230, 15, n_samples)  # Voltage in V
    current = np.random.normal(50, 10, n_samples)   # Current in A
    frequency = np.random.normal(50, 0.5, n_samples)  # Frequency in Hz
    power_factor = np.random.uniform(0.8, 1.0, n_samples)  # Power factor
    phase_angle = np.random.uniform(-30, 30, n_samples)  # Phase angle in degrees
    
    # Calculate load based on voltage and current with some noise
    load = (voltage * current * power_factor / 1000) + np.random.normal(0, 2, n_samples)  # Load in kW
    
    # Create stability based on realistic grid conditions
    stability_score = (
        (voltage / 230) * 0.3 +
        (frequency / 50) * 0.3 +
        power_factor * 0.2 +
        (1 - abs(phase_angle) / 30) * 0.2
    )
    
    # Add some randomness and create binary stability
    stability_score += np.random.normal(0, 0.1, n_samples)
    stability = (stability_score > 0.85).astype(int)
    
    # Introduce some anomalies (5% of data)
    anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    voltage[anomaly_indices] += np.random.normal(0, 50, len(anomaly_indices))
    current[anomaly_indices] += np.random.normal(0, 20, len(anomaly_indices))
    frequency[anomaly_indices] += np.random.normal(0, 2, len(anomaly_indices))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Voltage': voltage,
        'Current': current,
        'Frequency': frequency,
        'Power_Factor': power_factor,
        'Load': load,
        'Phase_Angle': phase_angle,
        'Stability': stability
    })
    
    # Ensure realistic bounds
    data['Voltage'] = np.clip(data['Voltage'], 180, 280)
    data['Current'] = np.clip(data['Current'], 20, 100)
    data['Frequency'] = np.clip(data['Frequency'], 48, 52)
    data['Power_Factor'] = np.clip(data['Power_Factor'], 0.7, 1.0)
    data['Load'] = np.clip(data['Load'], 5, 25)
    
    return data

def generate_time_series_data(n_days=30, samples_per_day=24, random_state=42):
    """
    Generate time series power grid data for load prediction
    
    Args:
        n_days (int): Number of days to generate
        samples_per_day (int): Samples per day (hourly data)
        random_state (int): Random seed
    
    Returns:
        pd.DataFrame: Time series dataset
    """
    np.random.seed(random_state)
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_days)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_days * samples_per_day)]
    
    # Generate load with daily and weekly patterns
    hours = np.array([ts.hour for ts in timestamps])
    days = np.array([ts.weekday() for ts in timestamps])
    
    # Daily pattern (higher during day, lower at night)
    daily_pattern = 10 + 8 * np.sin(2 * np.pi * hours / 24 + np.pi/2)
    
    # Weekly pattern (higher on weekdays)
    weekly_pattern = np.where(days < 5, 1.2, 0.8)
    
    # Base load with patterns and noise
    load = daily_pattern * weekly_pattern + np.random.normal(0, 1, len(timestamps))
    
    # Generate corresponding features
    voltage = 230 + np.random.normal(0, 10, len(timestamps))
    current = load * 2 + np.random.normal(0, 3, len(timestamps))
    frequency = 50 + np.random.normal(0, 0.3, len(timestamps))
    
    return pd.DataFrame({
        'Timestamp': timestamps,
        'Load': load,
        'Voltage': voltage,
        'Current': current,
        'Frequency': frequency
    })

if __name__ == "__main__":
    # Generate and save sample data
    data = generate_power_grid_data()
    data.to_csv('sample_power_grid_data.csv', index=False)
    print(f"Generated {len(data)} samples of power grid data")
    print(data.head())