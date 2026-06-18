import streamlit as st
import pandas as pd
import plotly.express as px
import os

def run_dashboard():
    st.set_page_config(page_title="Trend Analyzer Dashboard", layout="wide")
    st.title("Large-Scale Pedagogical Trend Analyzer")
    st.markdown("### Step 6: Reactive UI Rendering")
    st.markdown("This dashboard visualizes historical aggregations and predictive forecasts for academic sub-fields automatically detected via NLP and Community Detection.")

    data_file = 'trends_output.csv'
    
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        
        st.header("Temporal Topic Trends (Historical & Forecast)")
        
        # Interactive plot with plotly
        fig = px.line(
            df, 
            x='year', 
            y='share', 
            color='topic', 
            line_dash='is_forecast',
            markers=True,
            title="Topic Market Share Over Time (Dashed lines indicate Forecasts)",
            labels={"share": "Market Share", "year": "Year"}
        )
        # Update styling for forecast lines
        fig.update_layout(hovermode="x unified", xaxis_type="linear")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Raw Aggregated Data")
        st.dataframe(df.sort_values(['year', 'topic']))
    else:
        st.warning(f"Data file '{data_file}' not found. Please run main.py first to generate the trends.")

if __name__ == "__main__":
    run_dashboard()
