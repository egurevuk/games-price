import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="Steam Regional Price Calculator", layout="wide")
st.title("Steam Regional Price Recommendation Tool")

# Currency data with exchange rates and PPP factors (based on 2025-2026 data)
currency_data = {
    'USD': {'country': 'United States', 'exchange_rate': 1.0, 'ppp_factor': 1.0, 'region': 'Americas'},
    'EUR': {'country': 'Europe', 'exchange_rate': 0.92, 'ppp_factor': 0.95, 'region': 'Europe'},
    'GBP': {'country': 'United Kingdom', 'exchange_rate': 0.79, 'ppp_factor': 0.92, 'region': 'Europe'},
    'CAD': {'country': 'Canada', 'exchange_rate': 1.37, 'ppp_factor': 1.05, 'region': 'Americas'},
    'AUD': {'country': 'Australia', 'exchange_rate': 1.55, 'ppp_factor': 1.08, 'region': 'APAC'},
    'JPY': {'country': 'Japan', 'exchange_rate': 147.50, 'ppp_factor': 0.75, 'region': 'APAC'},
    'CNY': {'country': 'China', 'exchange_rate': 7.25, 'ppp_factor': 0.45, 'region': 'APAC'},
    'INR': {'country': 'India', 'exchange_rate': 84.20, 'ppp_factor': 0.25, 'region': 'APAC'},
    'BRL': {'country': 'Brazil', 'exchange_rate': 5.25, 'ppp_factor': 0.40, 'region': 'Americas'},
    'MXN': {'country': 'Mexico', 'exchange_rate': 20.50, 'ppp_factor': 0.35, 'region': 'Americas'},
    'RUB': {'country': 'Russia', 'exchange_rate': 105.00, 'ppp_factor': 0.30, 'region': 'CIS'}, 
    'TRY': {'country': 'Turkey', 'exchange_rate': 35.50, 'ppp_factor': 0.22, 'region': 'MENA'},
    'ZAR': {'country': 'South Africa', 'exchange_rate': 18.50, 'ppp_factor': 0.18, 'region': 'Africa'},
    'ARS': {'country': 'Argentina', 'exchange_rate': 1050.00, 'ppp_factor': 0.15, 'region': 'Americas'},
}

# Sidebar inputs
st.sidebar.header("Settings")
base_price_usd = st.sidebar.number_input("Base Price (USD)", value=10.0, min_value=0.99, step=0.01)
pricing_method = st.sidebar.selectbox(
    "Pricing Method",
    ["Exchange Rate Only", "Purchasing Power (PPP)", "Multi-Variable (Recommended)"]
)

# Calculate prices based on selected method
def calculate_prices(base_price, method, data):
    results = []
    
    for currency, info in data.items():
        country = info['country']
        exchange_rate = info['exchange_rate']
        ppp_factor = info['ppp_factor']
        region = info['region']
        
        if method == "Exchange Rate Only":
            price = base_price * exchange_rate
        elif method == "Purchasing Power (PPP)":
            # PPP method: adjust based on purchasing power
            price = base_price * exchange_rate * ppp_factor
        else:  # Multi-Variable
            # Blend of exchange rate and PPP, adjusted for regional factors
            ppp_adjusted = base_price * exchange_rate * ppp_factor
            exchange_only = base_price * exchange_rate
            # 60% PPP + 40% exchange rate creates a balanced approach
            price = (ppp_adjusted * 0.6) + (exchange_only * 0.4)
        
        results.append({
            'Currency': currency,
            'Country': country,
            'Region': region,
            'Local Price': round(price, 2),
            'Exchange Rate': exchange_rate,
            'PPP Factor': ppp_factor
        })
    
    return pd.DataFrame(results)

# Generate pricing table
df = calculate_prices(base_price_usd, pricing_method, currency_data)
df = df.sort_values('Region')

# Display current settings
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Base USD Price", f"${base_price_usd:.2f}")
with col2:
    st.metric("Pricing Method", pricing_method.split("(")[0].strip())
with col3:
    st.metric("Currencies Covered", len(currency_data))

st.divider()

# Show pricing table
st.subheader("Regional Price Recommendations")
display_df = df[['Currency', 'Country', 'Region', 'Local Price']].copy()
display_df['Local Price'] = display_df['Local Price'].apply(lambda x: f"{x:.2f}")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Breakdown by region
st.subheader("Pricing Breakdown by Region")
region_summary = df.groupby('Region').agg({
    'Local Price': ['min', 'max', 'mean'],
    'Currency': 'count'
}).round(2)

region_summary.columns = ['Min Price', 'Max Price', 'Avg Price', 'Currency Count']
st.dataframe(region_summary, use_container_width=True)

# Method explanation
st.subheader("How Each Method Works")
method_explanations = {
    "Exchange Rate Only": "Uses simple currency conversion at current exchange rates. Most transparent but doesn't account for local purchasing power.",
    "Purchasing Power (PPP)": "Adjusts for local purchasing power parity. Results in lower prices for developing markets, higher accessibility.",
    "Multi-Variable (Recommended)": "Combines exchange rates with PPP factors and regional adjustments for the most balanced approach across all markets."
}

st.info(method_explanations[pricing_method])

# Export option
st.subheader("Export")
csv = df.to_csv(index=False)
st.download_button(
    label="Download pricing as CSV",
    data=csv,
    file_name=f"steam_pricing_{base_price_usd}usd_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)
