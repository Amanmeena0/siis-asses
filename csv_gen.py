import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n_rows = 200

# Generate data
segments = ['VIP', 'Regular', 'New', 'Churn-Risk']
true_values = np.random.uniform(100, 5000, n_rows)

# Simulate Overfitting: 
# Predicted values are very accurate for the first 80% (simulating training data)
# and very inaccurate for the last 20% (simulating the test data)
predicted_values = true_values.copy()
# Add small noise to "train"
predicted_values[:160] += np.random.normal(0, 45, 160) 
# Add massive error to "test" (to match that 890 RMSE)
predicted_values[160:] += np.random.normal(800, 900, 40)

# Create DataFrame
df = pd.DataFrame({
    'customer_id': range(1001, 1001 + n_rows),
    'true_value': true_values,
    'predicted_value': np.clip(predicted_values, 0, None), # No negative CLV
    'customer_segment': np.random.choice(segments, n_rows),
    'transaction_date': pd.date_range(start='2024-01-01', periods=n_rows, freq='D')
})

# Save to CSV
df.to_csv('clv_predictions.csv', index=False)
print("File 'clv_predictions.csv' has been generated!")