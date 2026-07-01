import pandas as pd
import os

columns = [
    "Protocol",
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",

    "Packet Length Mean",
    "Packet Length Std",
    "Max Packet Length",
    "Min Packet Length",
    "Average Packet Size",

    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Header Length",

    "Total Backward Packets",
    "Total Length of Bwd Packets",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Header Length",

    "SYN Flag Count",
    "ACK Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "URG Flag Count",

    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Active Mean",
    "Idle Mean",

    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Down/Up Ratio",
    "binary_label"
]


def data_preprocessing():
    data= pd.read_csv(r"data/ingested_data.csv")
    updated_columns= data[columns]
    return updated_columns


def main():
    updated_data = data_preprocessing()
    print("Data preprocessing completed successfully!")
    print(updated_data.head())
    print(f"Data shape: {updated_data.shape}")
    updated_data.info()
    print(updated_data.describe())
    print(updated_data.isnull().sum())
    
    os.makedirs("data", exist_ok=True)
    updated_data.to_csv("data/preprocessed_data.csv", index=False)
    
if __name__ == "__main__":
    main()
