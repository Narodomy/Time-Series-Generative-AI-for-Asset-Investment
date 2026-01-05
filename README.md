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

## 🚀 Installation
You can set up the project environment using one of the two methods below. Method 1 (Conda) is recommended for better dependency management and reproducibility.

### 1. (Recommended) Using Conda
This method creates a complete, isolated environment using the exact package versions specified, including Python itself.
```
conda env create -f tsgen.yml
conda activate tsgen
```

### 2. Install the required packages
```pip install -r requirements.txt```

### 3. Extract the packages
```pip install -e .```

## Data Structure
### Design Class
#### 1. SingleAsset (Atomic Unit):
Manages individual asset data and handles its own technical indicators.
#### 2. AssetBasket (Data Aggregator):
Responsible for data loading, preprocessing, and aligning multiple assets to ensure consistency.
#### 3. Portfolio (Execution & Evaluation):
Portfolio shouldn't **inherit from** ```Dataset``` and ```AssetBasket``` classes, but it should be **Standalone class** to obtain the price data for calculation **NAV (Net Asset Value)**
1. **Concept: Separation of Concerns**
    * **AssetBasket (Data Layer):** Acts as a repository for historical price data (e.g., AAPL, GOOGL, TSLA over 10 years).
    * **Portfolio (Logic Layer):** Defines the asset allocation (e.g., 50/50 split) and calculates the portfolio's performance based on provided price paths.
2. **Code Structure (Pythonic Design)**
To facilitate benchmarking, the Portfolio is designed to process both Historical Data (```AssetBasket```) and **Generated Path Data.** This decoupled approach allows for direct comparison between real-world performance and model-driven simulations.
The final layer that manages investment strategies, performance calculation, and benchmarking.
