import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# ✅ Connect to PostgreSQL using SQLAlchemy
DATABASE_URL = "postgresql://postgres:aoufnihed@localhost/electric_db"  # Update with your database name
engine = create_engine(DATABASE_URL)

# ✅ Load data into Pandas DataFrame
try:
    df = pd.read_sql("SELECT * FROM electric_data", engine)
    st.write("📊 Live Electric Data")
    st.dataframe(df)
    
    # ✅ Check available columns
    st.write("**Columns in database:**", df.columns.tolist())
    
    # ✅ Ensure 'frequency' exists before using it
    columns_to_plot = ["voltage", "current", "load", "harmonics"]  # Default columns
    if "frequency" in df.columns:
        columns_to_plot.append("frequency")
    
    # ✅ Display line chart
    st.line_chart(df[columns_to_plot])
    
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
