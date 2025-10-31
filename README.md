# Time-Series-Generative-AI-for-Asset-Investment
Thesis project repository: Generative AI for time-series asset price simulation and investment strategy exploration.
Abstract 
A Fundamental challenge in investment risk management is that observed asset prices capture a 
single historical realization. This reliance on a single historical trajectory limits risk analysis, as it 
cannot capture uncertainty or account for situations that have never occurred before. Real-world 
examples such as Zillow Offers and Opendoor highlight the limitation of models in capturing anything 
beyond a single historical asset price trajectory. This research applies Generative AI to simulate 
synthetic asset price trajectories, with underlying contexts that include macroeconomic conditions, 
financial statements, historical prices, and news sentiment. The main objective is to construct 
scenarios that reflect the inherent uncertainty and volatility of financial markets. Intrinsic asset values 
are derived from fundamental data and then compared with observed prices to capture deviations 
driven by behavioral factors. Monte Carlo Simulation is employed to model distributions of potential 
intrinsic value paths, capturing uncertainty that extends beyond historical records. By integrating 
both fundamental and behavioral perspectives, the framework produces diverse and realistic 
datasets that enhance risk assessment and support more adaptive investment strategies.

🚀 Installation
You can set up the project environment using one of the two methods below. Method 1 (Conda) is recommended for better dependency management and reproducibility.

1. (Recommended) Using Conda
This method creates a complete, isolated environment using the exact package versions specified, including Python itself.

Bash

# 1. Create the new environment from the .yml file
conda env create -f my_data_env.yml

# 2. Activate the new environment
conda activate my_data_env
2. Using Pip
This method is suitable if you are not using Conda. It is highly recommended to use a virtual environment.

Bash

# 1. (Optional but recommended) Create a virtual environment
python -m venv venv

# 2. (Optional but recommended) Activate the virtual environment
# On macOS/Linux
source venv/bin/activate
# On Windows
.\venv\Scripts\activate

# 3. Install the required packages
pip install -r requirements.txt