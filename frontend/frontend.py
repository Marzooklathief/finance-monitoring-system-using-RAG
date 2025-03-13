import streamlit as st
import requests

st.title("📊 AI-Powered Financial Risk Monitoring")

if st.button("Get Live Risk Alerts"):
    try:
        response = requests.get("http://127.0.0.1:8000/alerts", timeout=5)
        response.raise_for_status()  # Ensure we only process successful responses

        alerts = response.json()
        if isinstance(alerts, list) and alerts:  # Ensure it's a list and not empty
            for alert in alerts:
                st.error(f"⚠️ Risk Alert for {alert.get('symbol', 'Unknown')}: Sudden drop in price!")
        else:
            st.info("✅ No risk alerts at the moment.")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch alerts: {str(e)}")

    except requests.exceptions.JSONDecodeError:
        st.error("❌ Received an invalid JSON response from the backend.")
