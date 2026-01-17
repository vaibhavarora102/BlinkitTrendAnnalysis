# Blinkit Trend Analysis & Sales Prediction

## Overview
This project performs a comprehensive trend analysis and sales forecasting for Blinkit. By integrating data from orders, customers, inventory, and customer feedback, it builds a machine learning pipeline to predict future sales revenue. The analysis identifies key drivers of revenue, such as pricing effects, stock efficiency, and customer sentiment.

## Project Structure
```
BlinkitTrendAnnalysis/
├── Notebooks/
│   └── SalesPrediction.ipynb   # Main notebook for EDA, Feature Engineering, and Modelling
├── trained_models/
│   └── xgb_model.pkl           # Trained XGBoost regressor
└── README.md                   # Project documentation
```
*Note: Datasets are expected to be in a `../datasets/` or `datasets/` directory relative to the notebook.*

## Methodology

### 1. Data Integration
The project merges multiple data sources to create a unified view:
- **Orders**: Transactional data.
- **Customers**: Demographics and segment info.
- **Feedback**: Sentiment analysis derived from customer feedback.
- **Inventory**: Stock availability and received quantities.
- **Order Items**: Item-level granularity.

### 2. Feature Engineering
Key features created for the model include:
- **Temporal Features**: Day, Month, Day of Week extracted from order dates.
- **Sentiment Scores**: Mapping textual feedback to numerical sentiment scores.
- **Rolling Statistics**: 7-day Rolling Mean and Standard Deviation of revenue to capture trends.
- **Efficiency Metrics**:
  - `stock_to_revenue_ratio`: Efficiency of stock utilization.
  - `revenue_growth_rate`: Week-over-week growth.
  - `price_effect_gap`: Difference between actual and expected revenue.

### 3. Machine Learning Model
- **Algorithm**: XGBoost Regressor (`XGBRegressor`).
- **Optimization**: Hyperparameter tuning using `GridSearchCV`.
- **Evaluation**: The model is evaluated using:
  - R-squared ($R^2$)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

## Getting Started

### Prerequisites
Ensure you have the following Python packages installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost category_encoders
```

### Running the Analysis
1. Clone the repository:
   ```bash
   git clone https://github.com/vaibhavarora102/BlinkitTrendAnnalysis.git
   ```
2. Navigate to the notebooks directory:
   ```bash
   cd BlinkitTrendAnnalysis/Notebooks
   ```
3. Open and run `SalesPrediction.ipynb` in Jupyter Notebook or your preferred IDE.

## Insights
The analysis explores correlations between stock levels, pricing strategies, and customer sentiment with total revenue, providing actionable insights for inventory management and sales strategy.
