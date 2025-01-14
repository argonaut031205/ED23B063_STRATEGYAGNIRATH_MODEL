# model for velocity optimization and Energy Management in Solar Cars with SOC estimation

This project models and optimizes the velocity of a solar-powered car to minimize power consumption while maintaining an efficient state of charge (SOC) for its battery. Using an Adam optimizer and incorporating solar dynamics, aerodynamics, and battery efficiency, the model aims to create an optimized velocity profile and analyze energy consumption metrics.

## Features
- **Velocity Optimization**: Uses the Adam optimizer to iteratively adjust velocity for minimal power consumption.
- **SOC Prediction**: Calculates and plots the battery's state of charge over time.
- **Energy Drain Analysis**: Evaluates energy consumption during different intervals.
- **Solar Panel Integration**: Considers solar irradiance and panel angle to calculate power gained from the sun.

## Technical Details
- **Programming Language**: Python
- **Libraries Used**:
  - `numpy` for numerical computations
  - `matplotlib` for data visualization
  - `random` for generating solar irradiance values
- **Optimization Algorithm**: Adam Optimizer with adaptive learning rates

## Assumptions

- The car is modeled with a mass of 300 kg, frontal area of 2 m², and an efficiency of 65%.
- Solar irradiance is simulated in the range of 1300–1400 W/m².
- The battery has a capacity of 4300 mAh at 150V.

## How It Works
1. **Inputs**: Solar irradiance, car mass, velocity data.
2. **Angle of Inclination**: Calculates the angle of inclination of solar panels using latitude, longitude, and declination angle.
3. **Energy**: Computes power drained from the battery and energy gained from solar panels.
4. **Optimization**: Iteratively adjusts velocity using the Adam optimizer to minimize power consumption.
5. **Visualization**: Generates plots for:
   - Initial vs. optimized velocity profiles
   - SOC over time
   - Energy drained over time

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/solar-car-optimization.git
   cd solar-car-optimization
