# Sales & Customer Segmentation Analytics System

A data analytics project that integrates business sales monitoring with unsupervised machine learning to track revenue performance and cluster customer purchasing behaviors.

## 📊 Business Context
This project serves as a combined solution for two enterprise management tracks:
1. **Sales & Revenue Analysis**: A real-time executive dashboard monitoring macro-economic trends, regional revenue generation, and product categories performance.
2. **Customer Segmentation Engine**: An algorithmic model classifying customer behavior profiles into structured categories (Recency, Frequency, Monetary metrics) via **K-Means Clustering**.

## 🛠️ Tech Stack & Requirements
* **Language:** Python 3.14+
* **Data Processing & ML:** Pandas, NumPy, Scikit-learn
* **Interactive Visualization Dashboard:** Dash (by Plotly), Plotly Express

To install all dependencies at once, run:
```bash
pip install pandas numpy scikit-learn plotly dash
```

## 🚀 Execution & Deployment

### 1. Data Ingestion & Transformation Pipeline
Before launching the web infrastructure, initialize the core business analytics pipeline to preprocess transactions and map client records to machine learning behavioral segments:
```bash
python data_generation.py
```
This produces a production-ready export dataset: `sales_customer_segments.csv`.

### 2. Launching the Interactive Web Dashboard
Run the primary web server application file:
```bash
python dashboard_app.py
```
Once initialized, navigate your web browser to: **`http://127.0.0`**

## 🧠 Algorithmic Framework
The client parsing engine normalizes consumer metrics via standard feature scaling and segregates customers into 3 behavioral profiles:
* 💎 **Loyal High-Spenders**: Top tier consumers characterized by high transaction frequency and high cash inflow.
* 🕒 **Casual Buyers**: Regular transactional group with steady baseline spending but wider order intervals.
* ⚠️ **At-Risk Customers**: Customers exhibiting lagging recency metrics that require targeted re-engagement strategies.
