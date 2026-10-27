"""Central configuration for the Factory Noise Monitoring simulation."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "noise_cnn.keras")
DB_PATH = os.path.join(BASE_DIR, "logs", "noise_log.db")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---- Audio / features ----
SAMPLE_RATE = 16000          # Hz
WINDOW_SEC = 1.0             # one analysis window = 1 second
N_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
N_MELS = 64
HOP_LENGTH = 125             # -> 16000/125 + 1 = 129 frames
N_FFT = 512
FMAX = 8000

# ---- Classes (order matters: index = CNN output neuron) ----
CLASSES = ["Normal", "Bearing fault", "Air leak", "Impact", "Overload"]
# Everything except "Normal" raises an anomaly alert
NORMAL_CLASS = "Normal"

# ---- Rules / alerts ----
NOISE_LIMIT_DBA = 85.0       # NIOSH recommended 8-hour exposure limit
WARNING_DBA = 80.0           # amber warning
CONFIDENCE_THRESHOLD = 0.80  # minimum CNN confidence to raise anomaly alert

# ---- Simulated factory nodes ----
# node_id -> (location, machine state schedule is random unless set via CLI)
NODES = {
    "NODE-01": "Press Shop",
    "NODE-02": "Compressor Room",
    "NODE-03": "Assembly Line",
}

# ---- MQTT (optional) ----
MQTT_HOST = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "factory/noise"
