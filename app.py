import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Live Sales Investment Advisor",
    layout="wide"
)

st.title("📈 Live Sales Forecast + Investment Decision System")

# -------------------------------------------------
# Sidebar inputs
# -------------------------------------------------
file = st.sidebar.file_uploader("Upload Sales CSV", type=["csv"])

days = st.sidebar.slider(
    "Days to Predict",
    7,
    90,
    30
)

investment_amount = st.sidebar.number_input(
    "💰 Investment Amount (₹)",
    min_value=0.0,
    value=10000.0,
    step=1000.0
)

profit_margin = st.sidebar.slider(
    "📊 Profit Margin (%)",
    1,
    50,
    20
)

# -------------------------------------------------
# Main App
# -------------------------------------------------
if file:

    # Read data
    df = pd.read_csv(file)

    # Rename columns for Prophet
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])

    st.subheader("📊 Raw Sales Data")
    st.dataframe(df, use_container_width=True)

    # -------------------------------------------------
    # Train Prophet Model
    # -------------------------------------------------
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Future prediction
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # -------------------------------------------------
    # Forecast Plot
    # -------------------------------------------------
    st.subheader("📈 Forecast")

    fig = model.plot(forecast)
    st.pyplot(fig)

    # -------------------------------------------------
    # Predicted Data
    # -------------------------------------------------
    predicted_df = forecast[["ds", "yhat"]].tail(days)
    predicted_df.columns = ["Date", "Predicted Sales"]

    st.subheader("📋 Predicted Sales")
    st.dataframe(predicted_df, use_container_width=True)

    # -------------------------------------------------
    # Investment Decision Logic
    # -------------------------------------------------
    st.subheader("🚦 Live Investment Decision")

    history_avg = df["y"].mean()
    future_avg = predicted_df["Predicted Sales"].mean()

    growth_rate = (future_avg - history_avg) / history_avg

    expected_profit = investment_amount * growth_rate * (profit_margin / 100)

    col1, col2, col3 = st.columns(3)

    col1.metric("Past Avg Sales", f"{history_avg:.2f}")
    col2.metric("Predicted Avg Sales", f"{future_avg:.2f}")
    col3.metric("Growth %", f"{growth_rate*100:.2f}%")

    st.metric("Expected Profit (₹)", f"{expected_profit:.2f}")

    # -------------------------------------------------
    # Final Decision
    # -------------------------------------------------
    if growth_rate > 0.05 and expected_profit > 0:
        st.success("✅ GOOD TIME TO INVEST")
        st.toast("Invest now — forecast strong 📈")
    else:
        st.error("❌ NOT SAFE TO INVEST NOW")
        st.toast("Wait — weak forecast ⚠️")

    # -------------------------------------------------
    # Reasoning
    # -------------------------------------------------
    st.markdown("### Reasoning")

    if growth_rate > 0.05:
        st.write("• Sales growth strong")
    else:
        st.write("• Sales growth weak")

    if expected_profit > 0:
        st.write("• Investment profitable")
    else:
        st.write("• Investment may lose money")

else:
    st.info("⬅ Upload CSV with columns: Date, Sales")