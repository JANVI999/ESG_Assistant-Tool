import pandas as pd
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Number of companies in the dataset
num_companies = 500                                                                                                                                                                            # Generate sample company names
company_names = [f'Company {i}' for i in range(1, num_companies + 1)]                                                                # Generate sample ESG data
data = []
for company in company_names:
    esg_score = random.uniform(1, 100)  # ESG score between 1 and 100
    environmental_score = random.uniform(1, 100)
    social_score = random.uniform(1, 100)
    governance_score = random.uniform(1, 100)
    stock_performance = random.uniform(-5, 5)  # Example 'Stock_Performance' column

    data.append([company, esg_score, environmental_score, social_score, governance_score, stock_performance])                                                                                                                                                                                # Create a DataFrame
columns = ['Company', 'ESG_Score', 'Environmental_Score', 'Social_Score', 'Governance_Score', 'Stock_Performance']
esg_df = pd.DataFrame(data, columns=columns)
# Save the dataset to a CSV file
esg_df.to_csv('large_esg_data_with_stock.csv', index=False)
# Display the first few rows of the dataset
print(esg_df.head())   
# Load the synthetic ESG dataset
esg_df = pd.read_csv('large_esg_data_with_stock.csv')                                                                                                   # Split the dataset into features (ESG scores) and target (e.g., stock performance)
X = esg_df[['ESG_Score', 'Environmental_Score', 'Social_Score', 'Governance_Score']]
y = esg_df['Stock_Performance']                                                                                                                                                # Initialize best_params
best_params = {}
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)                                   # Train a regression model to predict stock performance based on ESG scores
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model's performance (e.g., mean squared error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Function to get ESG recommendations
def get_esg_recommendation(company_name, esg_scores):
    prediction = model.predict([esg_scores])
    if prediction > 0:
        return f"Recommend investing in {company_name}."
    else:
        return f"Consider avoiding {company_name}."                                                                                                                # Example usage
company_name = 'Company 1'  # Replace with the company name you want to analyze
company_esg_scores = [70, 80, 75, 85]  # Replace with the actual ESG scores
recommendation = get_esg_recommendation(company_name, company_esg_scores)
print(recommendation)                                                                                                                                                                     # Risk Managment
# Assess potential risks such as overfitting and data bias
# Example: Evaluate overfitting using cross-validation scores
from sklearn.model_selection import cross_val_score
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
overfitting_risk = cross_val_scores.std()
print(f"Overfitting Risk (Std Dev of Cross-Val Scores): {overfitting_risk}")                                                                # Ethical Considerations:
# Ensure responsible use of AI in investment recommendations
# Example: Implement fairness and bias checks on model predictions
# (For simplicity, a basic example is provided here.)
predicted_returns = model.predict(X_test)
biased_predictions = [1 if pred > 0 else 0 for pred in predicted_returns]
ethical_concerns = sum(biased_predictions) / len(biased_predictions)
print(f"Ethical Concerns (Percentage of Positive Predictions): {ethical_concerns * 100}%")
# Model Accuracy:
# Regularly monitor and update the model to improve its accuracy
# Example: Track model performance and update hyperparameters
print(f"Best Model Hyperparameters: {best_params}")                                                                                                   from sklearn.metrics import r2_score

# Calculate R2 score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2:.2f}")      
