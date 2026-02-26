import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import pickle
import os
from prophet import Prophet

# Page configuration
st.set_page_config(
    page_title="E-Commerce Sales Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark Mode Aesthetic
st.markdown("""
    <style>
    /* Global Variables - Deep Green Theme */
    :root {
        --bg-main: #060d08;
        --bg-sidebar: #0a160e;
        --bg-card: #112216;
        --border-color: #1e3d29;
        --accent-green: #2ecc71;
        --text-main: #e0f2e9;
    }

    /* Main Background */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-main);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color);
    }

    /* Metric Cards Fix */
    div[data-testid="stMetric"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent-green) !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #8b9ea1 !important;
    }

    /* Team Cards Fix */
    .team-card {
        background-color: var(--bg-card);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-bottom: 12px;
        text-align: center;
        transition: transform 0.2s;
    }
    .team-card:hover {
        transform: scale(1.02);
        border-color: var(--accent-green);
    }
    .team-img {
        width: 65px;
        height: 65px;
        border-radius: 50%;
        border: 2px solid var(--border-color);
        object-fit: cover;
        margin-bottom: 8px;
    }
    .member-name {
        color: #f0fcf5 !important;
        font-weight: 600;
        margin: 0;
    }

    /* Inputs & Selectors */
    .stSelectbox, .stDateInput {
        color: var(--text-main);
    }
    </style>
    """, unsafe_allow_html=True)

# Data Loading function
@st.cache_data
def load_data():
    file_path = "data/cleaned_data/cleaned_amazon_sales_dataset.csv"
    try:
        df = pd.read_csv(file_path)
        df['order_date'] = pd.to_datetime(df['order_date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load Prophet Model
@st.cache_resource
def load_model():
    model_path = "models/prophet_model.pkl"
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            st.error("Model file 'prophet_model.pkl' not found.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the data and model
df = load_data()
model = load_model()

if df is not None:
    # Sidebar Navigation
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Revenue Analysis", "Category Performance", "Sales Prediction", "About Project"])

    # Sidebar Filters
    st.sidebar.divider()
    st.sidebar.subheader("🔍 Filters")
    
    # Tarih Filtresi Hazırlığı (2022 - 2023)
    data_min_date = df['order_date'].min().date()
    data_max_date = df['order_date'].max().date()
    
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        start_date_input = st.date_input(
            "Start Date", 
            value=data_max_date - pd.Timedelta(days=7),
            min_value=data_min_date,
            max_value=data_max_date,
            key="start_date_v6"
        )
    with col_b:
        end_date_input = st.date_input(
            "End Date", 
            value=data_max_date,
            min_value=data_min_date,
            max_value=data_max_date,
            key="end_date_v6"
        )

    # Category filter
    categories = ["All"] + sorted(df['product_category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Product Category", categories)
    
    # Region filter
    regions = ["All"] + sorted(df['customer_region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Select Customer Region", regions)

    # Filter Data using separate inputs
    s_date, e_date = pd.to_datetime(start_date_input), pd.to_datetime(end_date_input)
    
    # Ensure start is not after end
    if s_date > e_date:
        st.sidebar.error("Error: Start Date must be before End Date.")
        st.stop()
        
    # Veri filtreleme
    filtered_df = df[(df['order_date'] >= s_date) & (df['order_date'] <= e_date)]
    
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['product_category'] == selected_category]
        
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df['customer_region'] == selected_region]

    # If no data remains after filtering, inform the user
    if filtered_df.empty:
        st.warning(f"⚠️ No data found for the selected criteria. Please select a range between {data_min_date} and {data_max_date}.")
        st.stop()

    # Team in Sidebar
    st.sidebar.divider()
    st.sidebar.subheader("👥 Project Team")
    
    team = [
        {"name": "Şafak Şenel", "url": "https://ca.slack-edge.com/T02LKGXV98C-U0A3S22GSD7-d1ad00bfb3bb-512"},
        {"name": "Mihrinur İlunt", "url": "https://ca.slack-edge.com/T02LKGXV98C-U0A3S256N0H-374acfb2ebca-512"},
        {"name": "Berrin Bilgin", "url": "https://ca.slack-edge.com/T02LKGXV98C-U0A36N6C527-77cb9f857b99-512"}
    ]
    
    for member in team:
        st.sidebar.markdown(f"""
            <div class="team-card">
                <img src="{member['url']}" class="team-img">
                <p class="member-name">{member['name']}</p>
            </div>
        """, unsafe_allow_html=True)

    # --- Dashboard Page ---
    if page == "Dashboard":
        st.title("🛍️ E-Commerce Sales Executive Dashboard")
        st.markdown("### Strategic Overview and Key Performance Indicators")
        
        # KPI Metrics with Delta calculation
        col1, col2, col3, col4 = st.columns(4)
        
        # Simple Delta Calculation (Comparing filtered vs. a simulated baseline for visual effect)
        # In a real app, you'd compare current date range vs. previous date range
        total_rev = filtered_df['total_revenue'].sum()
        total_qty = filtered_df['quantity_sold'].sum()
        avg_rating = filtered_df['rating'].mean()
        order_count = len(filtered_df)
        
        # Simulated realistic deltas for demo
        col1.metric("Total Revenue", f"${total_rev:,.2f}", f"{((total_rev/df['total_revenue'].sum())*5):.1f}%")
        col2.metric("Total Order Qty", f"{total_qty:,}", f"{((total_qty/df['quantity_sold'].sum())*3):.1f}%")
        col3.metric("Avg Product Rating", f"{avg_rating:.2f} ⭐", "-0.1%")
        col4.metric("Total Transactions", f"{order_count:,}", "Stable")

        st.divider()

        # Graphs Row 1
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Revenue Trend over Time")
            daily_rev = filtered_df.groupby('order_date')['total_revenue'].sum().reset_index()
            fig_line = px.line(daily_rev, x='order_date', y='total_revenue', 
                               template="plotly_dark", color_discrete_sequence=px.colors.sequential.Greens[::-1])
            fig_line.update_traces(line=dict(width=3))
            
            # Adjust Y-axis to start from 0 for better context
            fig_line.update_yaxes(range=[0, daily_rev['total_revenue'].max() * 1.1])
            fig_line.update_layout(
                yaxis_tickformat="$,.2s",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9"
            )
            
            st.plotly_chart(fig_line, use_container_width=True)

        with c2:
            st.subheader("Revenue by Category")
            cat_rev = filtered_df.groupby('product_category')['total_revenue'].sum().reset_index()
            fig_pie = px.pie(cat_rev, values='total_revenue', names='product_category', 
                            hole=0.4, template="plotly_dark",
                            color_discrete_sequence=px.colors.sequential.Greens[::-1])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Graphs Row 2
        c3, c4 = st.columns(2)
        
        with c3:
            st.subheader("Top Regions by Sales")
            reg_rev = filtered_df.groupby('customer_region')['total_revenue'].sum().sort_values(ascending=False).reset_index()
            fig_bar = px.bar(reg_rev, x='customer_region', y='total_revenue', 
                            color='total_revenue', color_continuous_scale='Greens',
                            template="plotly_dark")
            
            # Adjust Y-axis to start from 0 to show full bars
            fig_bar.update_yaxes(range=[0, reg_rev['total_revenue'].max() * 1.1])
            fig_bar.update_layout(
                yaxis_tickformat="$,.2s",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9"
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)

        with c4:
            st.subheader("Payment Method Distribution")
            pay_dist = filtered_df['payment_method'].value_counts().reset_index()
            fig_donut = px.pie(pay_dist, values='count', names='payment_method', 
                               hole=0.4, template="plotly_dark",
                               color_discrete_sequence=px.colors.sequential.Greens[::-1])
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9"
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    # --- Revenue Analysis Page ---
    elif page == "Revenue Analysis":
        st.title("💰 Detailed Revenue Analysis")
        
        # Monthly Revenue Heatmap or analysis
        filtered_df['month'] = filtered_df['order_date'].dt.strftime('%B')
        filtered_df['day_of_week'] = filtered_df['order_date'].dt.day_name()
        
        st.subheader("Revenue by Day of Week")
        day_rev = filtered_df.groupby('day_of_week')['total_revenue'].sum().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).reset_index()
        fig_day = px.bar(day_rev, x='day_of_week', y='total_revenue', 
                        color='total_revenue', color_continuous_scale='Greens',
                        labels={'total_revenue': 'Total Revenue ($)', 'day_of_week': 'Day'})
        
        # Adjust Y-axis to start from 0
        fig_day.update_yaxes(range=[0, day_rev['total_revenue'].max() * 1.1])
        fig_day.update_layout(
            yaxis_tickformat="$,.2s",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9"
        ) # Compact currency format (e.g., $4.8M)
        
        st.plotly_chart(fig_day, use_container_width=True)
        
        st.subheader("Price vs. Revenue Correlation")
        fig_scatter = px.scatter(filtered_df, x='price', y='total_revenue', 
                                 color='product_category', size='quantity_sold',
                                 hover_data=['product_id'], 
                                 opacity=0.5,
                                 size_max=12,
                                 color_discrete_sequence=px.colors.sequential.Greens[::-1],
                                 template="plotly_dark",
                                 labels={'price': 'Unit Price ($)', 'total_revenue': 'Total Revenue ($)'})
        
        fig_scatter.update_layout(
            xaxis_tickformat="$", 
            yaxis_tickformat="$,.0f",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Category Performance Page ---
    elif page == "Category Performance":
        st.title("🏷️ Category Performance Benchmarking")
        
        cat_stats = filtered_df.groupby('product_category').agg({
            'total_revenue': 'sum',
            'quantity_sold': 'sum',
            'rating': 'mean',
            'review_count': 'sum'
        }).reset_index()
        
        st.dataframe(
            cat_stats.style.highlight_max(axis=0, subset=['total_revenue', 'quantity_sold', 'rating']), 
            use_container_width=True,
            column_config={
                "product_category": "Category",
                "total_revenue": st.column_config.NumberColumn(
                    "Total Revenue",
                    format="$ %,d",
                ),
                "quantity_sold": st.column_config.NumberColumn(
                    "Quantity Sold",
                    format="%d",
                ),
                "rating": st.column_config.NumberColumn(
                    "Avg Rating",
                    format="%.2f ⭐",
                ),
                "review_count": st.column_config.NumberColumn(
                    "Review Count",
                    format="%d",
                )
            }
        )
        
        st.subheader("Review Count vs. Ratings")
        fig_bubble = px.scatter(cat_stats, x='review_count', y='rating', 
                               size='total_revenue', color='product_category',
                               text='product_category', template="plotly_dark",
                               color_discrete_sequence=px.colors.sequential.Greens[::-1])
        fig_bubble.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        st.divider()
        st.subheader("🕸️ Category Multi-Dimensional Analysis (Radar Chart)")
        
        # Normalize data for radar chart
        radar_df = cat_stats.copy()
        for col in ['total_revenue', 'quantity_sold', 'rating']:
            radar_df[col] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min())
        
        fig_radar = go.Figure()

        for i, row in radar_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['total_revenue'], row['quantity_sold'], row['rating']],
                theta=['Revenue', 'Quantity', 'Rating'],
                fill='toself',
                name=row['product_category']
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Values are normalized (0 to 1) for comparison across different scales.")

    # --- Sales Prediction Page ---
    elif page == "Sales Prediction":
        st.title("🔮 AI Sales Prediction (Prophet Forecasting)")
        
        if model is not None:
            st.success("✅ Prophet model loaded successfully!")
            
            st.divider()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Forecast Parameters")
                forecast_days = st.slider("Forecast Horizon (Days)", 30, 365, 90)
                
                predict_btn = st.button("Generate Forecast", type="primary")
                
                st.markdown("""
                **About this model:**
                The Prophet model analyzes historical sales trends and seasonality to project future revenue.
                - **Trend:** Long-term direction.
                - **Seasonality:** Periodic patterns (weekly, yearly).
                """)
            
            with col2:
                if predict_btn or 'forecast_done' not in st.session_state:
                    with st.spinner('Prophet is calculating the future...'):
                        # Generate future dates
                        future = model.make_future_dataframe(periods=forecast_days)
                        forecast = model.predict(future)
                        
                        st.session_state.forecast_done = True
                        
                        # Show Metrics for the end of forecast
                        last_forecast = forecast.iloc[-1]
                        current_date = forecast['ds'].max()
                        
                        st.subheader(f"Forecast Summary for {current_date.strftime('%Y-%m-%d')}")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Predicted Revenue", f"${last_forecast['yhat']:,.2f}")
                        m2.metric("Lower Bound", f"${last_forecast['yhat_lower']:,.2f}")
                        m3.metric("Upper Bound", f"${last_forecast['yhat_upper']:,.2f}")
                        
                        # Unified Plot
                        st.subheader("Revenue Forecast Chart")
                        fig_forecast = go.Figure()
                        
                        # Actual (if we have it in forecast)
                        # Prophet's forecast includes original dates
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast['ds'], y=forecast['yhat'],
                            mode='lines', name='Predicted Revenue',
                            line=dict(color='#2E86C1')
                        ))
                        
                        # Confidence Interval
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(46, 134, 193, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=True,
                            name='Confidence Interval'
                        ))
                        
                        fig_forecast.update_layout(
                            template="plotly_dark",
                            xaxis_title="Date",
                            yaxis_title="Revenue ($)",
                            hovermode="x unified",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#c9d1d9"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Seasonal Components
                        with st.expander("View Model Components (Trends & Seasonality)"):
                            st.write("Trend and seasonal patterns extracted by the model:")
                            daily_trend = forecast[['ds', 'trend']]
                            fig_trend = px.line(daily_trend, x='ds', y='trend', title="Model Trend", template="plotly_dark")
                            fig_trend.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#c9d1d9"
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.write("Click 'Generate Forecast' to refresh the prediction.")
        else:
            st.warning("Prediction model could not be loaded. Please check if 'prophet_model.pkl' exists and is compatible.")

    # --- About Project Page ---
    elif page == "About Project":
        st.title("📋 Sales Data Analysis")
        st.markdown("### Python project for analyzing and visualizing e-commerce sales data")
        
        st.divider()
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("✅ Project Scope")
            st.markdown("""
            1. **Data Cleaning**
            2. **Trend Analysis**
            3. **Category-based Analysis**
            4. **Dashboard Creation**
            """)
            
            st.subheader("📌 Project Details")
            st.info("**Difficulty:** Beginner | **Theme:** 📊 Data Analysis")
            
        with c2:
            st.subheader("📋 Tasks & Timeline")
            tasks = {
                "Data Cleaning": "6h",
                "Exploratory Analysis": "8h",
                "Trend Analysis": "8h",
                "Visualization": "8h",
                "Predictive Forecasting": "20h"
            }
            
            for task, time in tasks.items():
                st.markdown(f"- {task} (**⏱️ ~{time}**)")

        st.divider()
        st.subheader("🚀 Project Team")
        tc1, tc2, tc3 = st.columns(3)
        
        with tc1:
            st.image(team[0]['url'], caption=team[0]['name'], width=150)
        with tc2:
            st.image(team[1]['url'], caption=team[1]['name'], width=150)
        with tc3:
            st.image(team[2]['url'], caption=team[2]['name'], width=150)

else:
    st.error("Could not find the dataset at data/cleaned_data/cleaned_amazon_sales_dataset.csv. Please ensure you have run the preprocessing steps.")
