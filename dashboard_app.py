import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# PHASE 1: DATA GENERATION & PREPROCESSING
# ==========================================
def generate_mock_data():
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
    df = pd.DataFrame(data)
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    return df

df_sales = generate_mock_data()

# ==========================================
# PHASE 2: CUSTOMER SEGMENTATION (PROJECT 2)
# ==========================================
def perform_segmentation(df):
    # Aggregate to RFM Metrics (Recency, Frequency, Monetary)
    latest_date = df['TransactionDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (latest_date - x.max()).days,
        'OrderID': 'count',
        'TotalSales': 'sum'
    }).rename(columns={'TransactionDate': 'Recency', 'OrderID': 'Frequency', 'TotalSales': 'Monetary'})
    
    # Scale Data & Apply K-Means
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Map Cluster IDs to Human Readable Segments
    cluster_mapping = {0: 'Casual Buyers', 1: 'Loyal High-Spenders', 2: 'At-Risk Customers'}
    rfm['Segment'] = rfm['Cluster'].map(cluster_mapping)
    
    return rfm[['Segment']]

# Merge Customer Segment Profiles back into Main Data
customer_segments = perform_segmentation(df_sales)
df_final = df_sales.merge(customer_segments, on='CustomerID', how='left')

# ==========================================
# PHASE 3: INTERACTIVE DASHBOARD (PROJECT 1)
# ==========================================
app = Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f8f9fa'}, children=[
    html.H1("Sales, Revenue & Customer Segmentation Portal", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # Global Filters Row
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '25px', 'justifyContent': 'center'}, children=[
        html.Div([
            html.Label("Filter by Region:"),
            dcc.Dropdown(id='region-drop', options=df_final['Region'].unique(), multi=True, placeholder="All Regions")
        ], style={'width': '300px'}),
        html.Div([
            html.Label("Filter by Customer Segment:"),
            dcc.Dropdown(id='segment-drop', options=df_final['Segment'].unique(), multi=True, placeholder="All Segments")
        ], style={'width': '300px'})
    ]),
    
    # Metrics Row
    html.Div(id='kpi-cards', style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '25px'}),
    
    # Visualizations Grid
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
        dcc.Graph(id='revenue-trend'),
        dcc.Graph(id='product-performance'),
        dcc.Graph(id='segment-distribution'),
        dcc.Graph(id='demographic-analysis')
    ])
])

@app.callback(
    [Output('kpi-cards', 'children'),
     Output('revenue-trend', 'figure'),
     Output('product-performance', 'figure'),
     Output('segment-distribution', 'figure'),
     Output('demographic-analysis', 'figure')],
    [Input('region-drop', 'value'),
     Input('segment-drop', 'value')]
)
def update_dashboard(selected_regions, selected_segments):
    filtered_df = df_final.copy()
    
    if selected_regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]
    if selected_segments:
        filtered_df = filtered_df[filtered_df['Segment'].isin(selected_segments)]
        
    # 1. Compute Dashboard KPIs
    total_rev = filtered_df['TotalSales'].sum()
    total_units = filtered_df['Quantity'].sum()
    avg_order = filtered_df['TotalSales'].mean() if not filtered_df.empty else 0
    
    kpi_html = [
        html.Div([html.H3("Total Revenue"), html.H2(f"${total_rev:,.2f}")], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center', 'width': '25%'}),
        html.Div([html.H3("Units Sold"), html.H2(f"{total_units:,}")], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center', 'width': '25%'}),
        html.Div([html.H3("Avg. Order Value"), html.H2(f"${avg_order:,.2f}")], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center', 'width': '25%'})
    ]
    
    # 2. Revenue Trend Line Chart
    trend_data = filtered_df.resample('D', on='TransactionDate')['TotalSales'].sum().reset_index()
    fig_trend = px.line(trend_data, x='TransactionDate', y='TotalSales', title='Revenue Trends Over Time', labels={'TotalSales': 'Revenue ($)'}, template='plotly_white')
    
    # 3. Product Performance Bar Chart
    prod_data = filtered_df.groupby('ProductCategory')['TotalSales'].sum().reset_index()
    fig_prod = px.bar(prod_data, x='ProductCategory', y='TotalSales', title='Revenue by Product Category', color='ProductCategory', template='plotly_white')
    
    # 4. Customer Segment Distribution Pie Chart
    seg_data = filtered_df.drop_duplicates(subset=['CustomerID']).groupby('Segment').size().reset_index(name='Count')
    fig_seg = px.pie(seg_data, values='Count', names='Segment', title='Customer Segment Share', hole=0.4, template='plotly_white')
    
    # 5. Age Demographics Box Plot
    fig_demo = px.box(filtered_df.drop_duplicates(subset=['CustomerID']), x='Segment', y='Age', title='Age Demographics by Segment', color='Segment', template='plotly_white')
    
    return kpi_html, fig_trend, fig_prod, fig_seg, fig_demo

if __name__ == '__main__':
    app.run(debug=True, port=8050)
