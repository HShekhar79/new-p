import pandas as pd
import joblib

def main():
    # Load the trained model
    model = joblib.load('models/rf_model.pkl')
    print("Model loaded for batch prediction.")

    # Load your new network flows CSV (replace with your file path)
    input_csv = 'data/new_network_flows.csv'  # Change this to your actual data file path
    df = pd.read_csv(input_csv)
    print(f"Loaded {df.shape[0]} rows for prediction from {input_csv}")

    # Select feature columns for prediction (same as training features)
    feature_cols = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Flow Bytes/s'
    ]
    
    # Clean column names in case of spaces
    df.columns = df.columns.str.strip()

    features = df[feature_cols]

    # Make predictions
    predictions = model.predict(features)

    # Append predictions to dataframe
    df['Predicted_Label'] = predictions

    # Save predictions to CSV
    output_csv = 'data/batch_predictions.csv'
    df.to_csv(output_csv, index=False)
    print(f"Batch predictions saved to {output_csv}.")

if __name__ == '__main__':
    main()
