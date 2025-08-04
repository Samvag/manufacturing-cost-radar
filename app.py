import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Cost Radar", page_icon="🎯", layout="wide")

st.title("🎯 Manufacturing Cost Leakage Radar")

# Sample data
data = {
    'Material Waste': 75,
    'Equipment Downtime': 60, 
    'Labor Inefficiency': 40,
    'Energy Consumption': 85,
    'Quality Defects': 30,
    'Inventory Excess': 65,
    'Process Variance': 70,
    'Supply Chain': 25
}

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # Use actual data if available
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 3:
        st.success(f"Found {len(numeric_cols)} numeric columns")
        # Update data based on file
        for i, col in enumerate(list(data.keys())[:len(numeric_cols)]):
            data[col] = min(90, max(10, df[numeric_cols[i]].mean() / 1000))

# Create radar chart
categories = list(data.keys())
values = list(data.values())

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='Cost Leakage %'
))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    showlegend=True,
    title="Cost Leakage Radar"
)

st.plotly_chart(fig, use_container_width=True)

# Summary
st.subheader("Summary")
for category, value in data.items():
    color = "🔴" if value >= 70 else "🟡" if value >= 40 else "🟢"
    st.write(f"{color} {category}: {value}%")
