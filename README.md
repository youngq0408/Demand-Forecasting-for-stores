# Demand Forecasting for Stores

## Project Overview
This project focuses on predicting product demand in retail stores using historical sales data. The data includes information about store IDs, SKU IDs, pricing, promotions, and weekly sales volume. A machine learning pipeline was built to preprocess the data, train a predictive model, and evaluate its performance using error metrics and visualization techniques.

## Objective
The main objective is to **Build a predictive model that accurately esitmates the number of units sold for each product in each store**

This helps businesses:
1. Forecast future inventory needs
2. Reduce overstock and stockouts 
3. Improve decision-making in pricing and promotions 

## Dataset
This project uses one datasets, `store_demand_forecasting.csv`
This dataset contains **150,150 rows** and **9 columns** 

- `record_ID:` Unique identifier for each transaction record.
- `week:` Date of the record (format: yy/mm/dd).
- `store_id:` Identifier for the store.
- `sku_id:`  Identifier for the product (SKU = Stock Keeping Unit).
- `total_price:`  Final price paid (could include discounts/promotions).
- `base_price:` Original price of the item.
- `is_featured_sku:` Binary flag (1/0) indicating if the product was featured.
- `is_display_sku:` Binary flag (1/0) indicating if the product was placed in a display area.
- `units_sold:` Target variable — number of units sold during the week.

## Step-by-Step Process

### **Step 1: Data Loading and Preprocessing**
Loaded the dataset using pandas
Extracted day, month and year from the week column
Dropped the original week column to clean the data

### **Step 2: Model Preparation**
Split the dataset into features (X) and target (y = units_sold).
Used train_test_split() to divide the data into training and testing sets.
Initialized and trained a Random Forest Regressor using the training data.

### **Step 3: Evaluation with RMSE**
Made predictions on the test set.
Evaluated model performance using Root Mean Squared Error (RMSE) and R² score.

### **Step 4: Visualization and Stats**
Used describe() and hist() to understand the distribution of units_sold.
Plotted predicted vs actual values to visually assess prediction accuracy.
Explored histograms for all features using matplotlib.

### **Step 5: Data Cleaning & Feature Engineering**
Dropped record_ID (non-informative column).
Checked for outliers using the 99th percentile of units_sold.
Removed extreme outliers to improve model performance.

### **Step 6: One-Hot Encoding**
Converted categorical columns store_id and sku_id into binary (dummy) variables using pd.get_dummies().

### **Step 7: Retrain Model After Encoding**
Repeated the model training using the updated dataset with one-hot encoded features.
Evaluated model performance again using R² and RMSE.

### **Step 8: Hyperparameter Tuning**
Used GridSearchCV to find the best combination of n_estimators and min_samples_split for the Random Forest model.
Selected the best model and evaluated it on the test set.


## Tools and Libraries
- **Python**: Core language used for analysis and modeling.
- **Pandas**: For data manipulation and preprocessing.
- **Numpy**:  For numerical operations and creating ranges used in plotting.
- **Scikit-Learn**: For machine learning modeling and evaluation.
- **Matplotlib**: For plotting basic visualizations.

## Results
- **Root Mean Squared Error(RMSE)**: RMSE (Root Mean Squared Error): Gave an interpretable measure of average prediction error.
- **R-squared(R^2)**: Provided a basic measure of how well the model explained variance in units_sold.
## Future Work
