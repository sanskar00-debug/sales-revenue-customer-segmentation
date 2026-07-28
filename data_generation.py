import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Generate the data
np.random.seed(42)
records = 1000
data = {
    'OrderID': np.random.randint(10000, 99999, size=records),
    'CustomerID': np.random.randint(100, 250, size=records),
    'TransactionDate': pd.date_range(start='2026-01-01', periods=records, freq='h'),
    'Age': np.random.randint(18, 70, size=records),
    'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], size=records, p=[0.48, 0.48, 0.04]),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], size=records),
    'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty'], size=records),
    'Quantity': np.random.randint(1, 6, size=records),
    'UnitPrice': np.random.uniform(10.0, 500.0, size=records).round(2)
}
df_sales = pd.DataFrame(data)
df_sales['TotalSales'] = df_sales['Quantity'] * df_sales['UnitPrice']

# 2. Run customer segmentation
latest_date = df_sales['TransactionDate'].max()
rfm = df_sales.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (latest_date - x.max()).days,
    'OrderID': 'count',
    'TotalSales': 'sum'
}).rename(columns={'TransactionDate': 'Recency', 'OrderID': 'Frequency', 'TotalSales': 'Monetary'})

scaler = StandardScaler()
scaled_features = scaler.fit_transform(rfm)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(scaled_features)

cluster_mapping = {0: 'Casual Buyers', 1: 'Loyal High-Spenders', 2: 'At-Risk Customers'}
rfm['Segment'] = rfm['Cluster'].map(cluster_mapping)

df_final = df_sales.merge(rfm[['Segment']], on='CustomerID', how='left')

# 3. Export cleanly
submission_columns = [
    'OrderID', 'CustomerID', 'TransactionDate', 'Age', 'Gender', 
    'Region', 'ProductCategory', 'Quantity', 'UnitPrice', 'TotalSales', 'Segment'
]
export_df = df_final[submission_columns].sort_values(by='TransactionDate')

# Save directly to your Desktop folder
file_path = "C:/Users/intel/Desktop/Data Project/Final_Sales_and_Segmentation_Data.csv"
export_df.to_csv(file_path, index=False)

print(f"Success! Check your folder. The file has been created at:\n{file_path}")
