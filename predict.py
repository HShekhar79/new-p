import joblib
import pandas as pd

# Load your saved model
model = joblib.load('models/rf_model.pkl')
print("Model loaded!")

# Take new network flow data (replace placeholder with actual data)
# Here, just copy a row of features as an example
# Make sure these show your feature columns (use same order as training)
new_data = {
    'Flow Duration': 1000,
    'Total Fwd Packets': 50,
    'Total Backward Packets': 50,
    'Total Length of Fwd Packets': 5000,
    'Total Length of Bwd Packets': 4800,
    'Fwd Packet Length Max': 1500,
    'Bwd Packet Length Max': 1500,
    'Flow Bytes/s': 300
}

# Convert into DataFrame (just one row for prediction)
df_new = pd.DataFrame([new_data])

# Make prediction
prediction = model.predict(df_new)
print("Predicted class:", prediction[0])
