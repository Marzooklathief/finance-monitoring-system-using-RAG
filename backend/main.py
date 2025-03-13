from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

# ✅ Get Latest Risk Alerts from Pathway
@app.get("/alerts")
def get_risk_alerts():
    try:
        response = requests.get("http://localhost:3000", timeout=5)  # Add timeout
        response.raise_for_status()  # Raise an error if status code is not 200

        # Try to parse JSON
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid JSON response from Pathway server.")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Pathway: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
