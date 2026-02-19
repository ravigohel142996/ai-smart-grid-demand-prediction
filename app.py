"""
AI Smart Grid Demand Prediction & Blackout Prevention System
Enterprise Grid Monitoring Platform powered by Machine Learning
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Smart Grid Demand Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">⚡ AI Smart Grid Demand Prediction & Blackout Prevention System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Machine Learning | Enterprise Grid Monitoring Platform</div>', unsafe_allow_html=True)
st.markdown("---")

# Functions for data generation and model training
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic smart grid dataset"""
    np.random.seed(42)
    
    # Generate features
    hour = np.random.randint(0, 24, n_samples)
    temperature = np.random.uniform(0, 50, n_samples)
    population = np.random.uniform(100, 1000, n_samples)
    industrial_load = np.random.uniform(0, 100, n_samples)
    renewable_energy = np.random.uniform(0, 100, n_samples)
    
    # Create realistic demand formula
    # Demand increases with hour (peak during day), temperature (AC usage),
    # population, and industrial load, but decreases with renewable energy
    demand = (
        50 * hour +  # Hour effect
        80 * temperature +  # Temperature effect (AC usage)
        5 * population +  # Population effect
        40 * industrial_load +  # Industrial load effect
        -30 * renewable_energy +  # Renewable energy reduces demand from grid
        np.random.normal(0, 500, n_samples)  # Random noise
    )
    
    # Ensure demand is positive
    demand = np.maximum(demand, 500)
    
    # Create DataFrame
    df = pd.DataFrame({
        'hour': hour,
        'temperature': temperature,
        'population': population,
        'industrial_load': industrial_load,
        'renewable_energy': renewable_energy,
        'demand': demand
    })
    
    return df

@st.cache_resource
def train_model(df):
    """Train the machine learning model"""
    X = df[['hour', 'temperature', 'population', 'industrial_load', 'renewable_energy']]
    y = df['demand']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, r2, mae

def predict_demand(model, hour, temperature, population, industrial_load, renewable_energy):
    """Predict electricity demand"""
    features = np.array([[hour, temperature, population, industrial_load, renewable_energy]])
    prediction = model.predict(features)[0]
    return max(prediction, 0)  # Ensure non-negative

def get_grid_status(demand):
    """Determine grid status based on demand"""
    if demand < 5000:
        return "Stable", "success"
    elif 5000 <= demand <= 8000:
        return "Moderate Risk", "warning"
    else:
        return "High Blackout Risk", "error"

def create_visualization(df, x_col, y_col, title, xlabel, ylabel):
    """Create matplotlib visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_col], df[y_col], alpha=0.5, c='#1f77b4')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=2, label='Trend')
    ax.legend()
    
    plt.tight_layout()
    return fig

# Generate data and train model
df = generate_synthetic_data(1000)
model, X_train, X_test, y_train, y_test, r2, mae = train_model(df)

# Sidebar - Navigation and Input Parameters
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard Overview", "Demand Prediction", "Grid Analytics", "Risk Monitoring", "AI Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Input Parameters")

# User inputs
hour = st.sidebar.slider("Hour of Day", 0, 23, 12, help="Select hour (0-23)")
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25, help="Ambient temperature in Celsius")
population = st.sidebar.slider("Population Load (thousands)", 100, 1000, 500, help="Population in thousands")
industrial_load = st.sidebar.slider("Industrial Load (%)", 0, 100, 50, help="Industrial load percentage")
renewable_energy = st.sidebar.slider("Renewable Energy (%)", 0, 100, 30, help="Renewable energy contribution")

st.sidebar.markdown("---")
st.sidebar.info("💡 Adjust parameters to see real-time predictions")

# Make prediction
predicted_demand = predict_demand(model, hour, temperature, population, industrial_load, renewable_energy)
grid_status, status_type = get_grid_status(predicted_demand)

# Calculate statistics
avg_demand = df['demand'].mean()
max_demand = df['demand'].max()
min_demand = df['demand'].min()

# Page routing
if page == "Dashboard Overview":
    st.header("📊 Dashboard Overview")
    st.markdown("Real-time monitoring of grid performance and demand prediction")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Predicted Demand",
            value=f"{predicted_demand:.0f} MW",
            delta=f"{predicted_demand - avg_demand:.0f} MW vs avg"
        )
    
    with col2:
        st.metric(
            label="Average Grid Demand",
            value=f"{avg_demand:.0f} MW"
        )
    
    with col3:
        st.metric(
            label="Peak Demand",
            value=f"{max_demand:.0f} MW"
        )
    
    with col4:
        st.metric(
            label="Min Demand",
            value=f"{min_demand:.0f} MW"
        )
    
    st.markdown("---")
    
    # Grid status
    st.subheader("⚡ Grid Status")
    if status_type == "success":
        st.success(f"✅ Grid Status: **{grid_status}** | Demand: {predicted_demand:.0f} MW")
    elif status_type == "warning":
        st.warning(f"⚠️ Grid Status: **{grid_status}** | Demand: {predicted_demand:.0f} MW")
    else:
        st.error(f"🚨 Grid Status: **{grid_status}** | Demand: {predicted_demand:.0f} MW")
    
    st.markdown("---")
    
    # Quick stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Current Parameters")
        st.write(f"**Hour:** {hour}:00")
        st.write(f"**Temperature:** {temperature}°C")
        st.write(f"**Population Load:** {population}k")
        st.write(f"**Industrial Load:** {industrial_load}%")
        st.write(f"**Renewable Energy:** {renewable_energy}%")
    
    with col2:
        st.subheader("🎯 Risk Assessment")
        st.write("**Thresholds:**")
        st.write("• Stable: < 5,000 MW")
        st.write("• Moderate Risk: 5,000 - 8,000 MW")
        st.write("• High Risk: > 8,000 MW")
        st.write("")
        st.write(f"**Current Status:** {grid_status}")

elif page == "Demand Prediction":
    st.header("🔮 Demand Prediction")
    st.markdown("AI-powered electricity demand forecasting")
    
    # Prediction result
    st.subheader("Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Demand",
            value=f"{predicted_demand:.0f} MW"
        )
    
    with col2:
        st.metric(
            label="Grid Status",
            value=grid_status
        )
    
    with col3:
        risk_percentage = min(100, (predicted_demand / 10000) * 100)
        st.metric(
            label="Risk Level",
            value=f"{risk_percentage:.1f}%"
        )
    
    st.markdown("---")
    
    # Status alert
    st.subheader("⚡ Grid Status Alert")
    if status_type == "success":
        st.success(f"✅ **{grid_status}** - The grid is operating within safe parameters. Current demand: {predicted_demand:.0f} MW")
    elif status_type == "warning":
        st.warning(f"⚠️ **{grid_status}** - The grid is experiencing elevated demand. Monitor closely. Current demand: {predicted_demand:.0f} MW")
    else:
        st.error(f"🚨 **{grid_status}** - Critical demand levels detected! Immediate action required. Current demand: {predicted_demand:.0f} MW")
    
    st.markdown("---")
    
    # Input summary
    st.subheader("📋 Input Summary")
    input_df = pd.DataFrame({
        'Parameter': ['Hour', 'Temperature', 'Population', 'Industrial Load', 'Renewable Energy'],
        'Value': [f"{hour}:00", f"{temperature}°C", f"{population}k", f"{industrial_load}%", f"{renewable_energy}%"]
    })
    st.dataframe(input_df, use_container_width=True)

elif page == "Grid Analytics":
    st.header("📊 Grid Analytics")
    st.markdown("Comprehensive analysis of demand patterns")
    
    # Create visualizations
    st.subheader("📈 Demand Analysis Visualizations")
    
    # Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = create_visualization(df, 'hour', 'demand', 
                                   'Electricity Demand vs Hour of Day',
                                   'Hour of Day', 'Demand (MW)')
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        fig2 = create_visualization(df, 'temperature', 'demand',
                                   'Electricity Demand vs Temperature',
                                   'Temperature (°C)', 'Demand (MW)')
        st.pyplot(fig2)
        plt.close()
    
    # Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = create_visualization(df, 'population', 'demand',
                                   'Electricity Demand vs Population Load',
                                   'Population (thousands)', 'Demand (MW)')
        st.pyplot(fig3)
        plt.close()
    
    with col2:
        fig4 = create_visualization(df, 'industrial_load', 'demand',
                                   'Electricity Demand vs Industrial Load',
                                   'Industrial Load (%)', 'Demand (MW)')
        st.pyplot(fig4)
        plt.close()
    
    st.markdown("---")
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Demand Statistics**")
        st.write(f"Mean: {df['demand'].mean():.0f} MW")
        st.write(f"Median: {df['demand'].median():.0f} MW")
        st.write(f"Std Dev: {df['demand'].std():.0f} MW")
    
    with col2:
        st.write("**Temperature Statistics**")
        st.write(f"Mean: {df['temperature'].mean():.1f}°C")
        st.write(f"Min: {df['temperature'].min():.1f}°C")
        st.write(f"Max: {df['temperature'].max():.1f}°C")
    
    with col3:
        st.write("**Population Statistics**")
        st.write(f"Mean: {df['population'].mean():.0f}k")
        st.write(f"Min: {df['population'].min():.0f}k")
        st.write(f"Max: {df['population'].max():.0f}k")

elif page == "Risk Monitoring":
    st.header("🚨 Risk Monitoring")
    st.markdown("Real-time blackout risk assessment and monitoring")
    
    # Risk level display
    st.subheader("⚡ Current Risk Level")
    
    if status_type == "success":
        st.success(f"### ✅ {grid_status}")
        st.write(f"**Current Demand:** {predicted_demand:.0f} MW")
        st.write(f"**Status:** The power grid is operating within safe parameters.")
        st.write(f"**Action Required:** Continue monitoring.")
    elif status_type == "warning":
        st.warning(f"### ⚠️ {grid_status}")
        st.write(f"**Current Demand:** {predicted_demand:.0f} MW")
        st.write(f"**Status:** The power grid is experiencing elevated demand levels.")
        st.write(f"**Action Required:** Prepare contingency measures. Monitor continuously.")
    else:
        st.error(f"### 🚨 {grid_status}")
        st.write(f"**Current Demand:** {predicted_demand:.0f} MW")
        st.write(f"**Status:** Critical demand levels detected! Risk of blackout is high.")
        st.write(f"**Action Required:** Activate emergency protocols. Implement load shedding if necessary.")
    
    st.markdown("---")
    
    # Risk thresholds
    st.subheader("📊 Risk Threshold Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Stable Zone**\n\n< 5,000 MW\n\nNormal operations")
    
    with col2:
        st.warning("**Moderate Risk**\n\n5,000 - 8,000 MW\n\nHeightened monitoring")
    
    with col3:
        st.error("**High Risk**\n\n> 8,000 MW\n\nEmergency protocols")
    
    st.markdown("---")
    
    # Risk factors
    st.subheader("⚠️ Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**High Risk Indicators:**")
        if hour >= 10 and hour <= 20:
            st.write("🔴 Peak hours (10:00 - 20:00)")
        else:
            st.write("🟢 Off-peak hours")
        
        if temperature > 35:
            st.write("🔴 High temperature (AC load)")
        elif temperature > 25:
            st.write("🟡 Moderate temperature")
        else:
            st.write("🟢 Low temperature")
        
        if population > 750:
            st.write("🔴 High population load")
        elif population > 500:
            st.write("🟡 Moderate population load")
        else:
            st.write("🟢 Low population load")
    
    with col2:
        st.write("**Mitigation Factors:**")
        if renewable_energy > 60:
            st.write("🟢 High renewable energy contribution")
        elif renewable_energy > 30:
            st.write("🟡 Moderate renewable energy")
        else:
            st.write("🔴 Low renewable energy")
        
        if industrial_load < 40:
            st.write("🟢 Low industrial load")
        elif industrial_load < 70:
            st.write("🟡 Moderate industrial load")
        else:
            st.write("🔴 High industrial load")

elif page == "AI Model Insights":
    st.header("🤖 AI Model Insights")
    st.markdown("Understanding the machine learning model and predictions")
    
    # Model performance
    st.subheader("📈 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="R² Score",
            value=f"{r2:.4f}",
            help="Coefficient of determination (1.0 is perfect)"
        )
    
    with col2:
        st.metric(
            label="Mean Absolute Error",
            value=f"{mae:.0f} MW",
            help="Average prediction error"
        )
    
    with col3:
        accuracy_percentage = r2 * 100
        st.metric(
            label="Model Accuracy",
            value=f"{accuracy_percentage:.2f}%",
            help="Overall model accuracy"
        )
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    
    st.write("The model uses the following features to predict electricity demand:")
    
    # Get model coefficients
    coefficients = model.coef_
    features = ['Hour', 'Temperature', 'Population', 'Industrial Load', 'Renewable Energy']
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Display feature importance
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.dataframe(
            feature_importance[['Feature', 'Coefficient']].reset_index(drop=True),
            use_container_width=True
        )
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in feature_importance['Coefficient']]
        ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
        ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax.set_title('Feature Impact on Demand Prediction', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Model explanation
    st.subheader("💡 Prediction Explanation")
    
    st.write("**How the model predicts demand:**")
    
    st.write(f"""
    Based on your current inputs:
    
    - **Hour ({hour}:00)**: Contributes {coefficients[0] * hour:.0f} MW to demand
    - **Temperature ({temperature}°C)**: Contributes {coefficients[1] * temperature:.0f} MW to demand
    - **Population ({population}k)**: Contributes {coefficients[2] * population:.0f} MW to demand
    - **Industrial Load ({industrial_load}%)**: Contributes {coefficients[3] * industrial_load:.0f} MW to demand
    - **Renewable Energy ({renewable_energy}%)**: Contributes {coefficients[4] * renewable_energy:.0f} MW to demand
    
    **Total Predicted Demand:** {predicted_demand:.0f} MW
    """)
    
    st.info("""
    **Model Type:** Linear Regression
    
    **Training Data:** 1,000 synthetic smart grid samples
    
    **Features:** Hour, Temperature, Population Load, Industrial Load, Renewable Energy
    
    **Target:** Electricity Demand (MW)
    
    **Use Case:** Enterprise power grid monitoring and blackout prevention
    """)
    
    st.markdown("---")
    
    # Model insights
    st.subheader("🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Positive Impact Factors:**")
        positive_features = feature_importance[feature_importance['Coefficient'] > 0]['Feature'].tolist()
        for f in positive_features:
            st.write(f"✅ {f} - Increases demand")
    
    with col2:
        st.write("**Negative Impact Factors:**")
        negative_features = feature_importance[feature_importance['Coefficient'] < 0]['Feature'].tolist()
        for f in negative_features:
            st.write(f"✅ {f} - Reduces demand")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>⚡ AI Smart Grid Demand Prediction & Blackout Prevention System</strong></p>
    <p>Enterprise Grid Monitoring Platform | Powered by Machine Learning</p>
    <p>© 2026 Smart Grid AI Solutions | Developed for Power Companies & Government Energy Departments</p>
</div>
""", unsafe_allow_html=True)
