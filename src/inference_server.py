"""Edge/cloud server: consumes node messages, runs the CNN, applies rules,
stores everything in SQLite and prints alerts."""
import os, sqlite3, threading
import numpy as np
import tensorflow as tf
import config
from src.features import log_mel


class InferenceServer(threading.Thread):
    def __init__(self, transport, model_path=config.MODEL_PATH):
        super().__init__(daemon=True)
        self.transport = transport
        self.model = tf.keras.models.load_model(model_path)
        self.running = True
        os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
        self.db = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self.db.execute("DROP TABLE IF EXISTS readings")
        self.db.execute("""CREATE TABLE readings(
            ts TEXT, node_id TEXT, location TEXT, level_dba REAL,
            predicted TEXT, confidence REAL, true_state TEXT,
            exposure_alert INTEGER, anomaly_alert INTEGER)""")
        self.db.commit()
        self.alerts = []

    def classify(self, audio):
        probs = self.model.predict(log_mel(audio)[np.newaxis], verbose=0)[0]
        i = int(np.argmax(probs))
        return config.CLASSES[i], float(probs[i])

    def run(self):
        while self.running:
            msg = self.transport.get(timeout=0.5)
            if msg is None:
                continue
            audio = np.asarray(msg["audio"], dtype=np.float32)
            label, conf = self.classify(audio)
            level = msg["level_dba"]
            exposure = int(level >= config.NOISE_LIMIT_DBA)
            anomaly = int(label != config.NORMAL_CLASS and conf >= config.CONFIDENCE_THRESHOLD)
            self.db.execute("INSERT INTO readings VALUES (?,?,?,?,?,?,?,?,?)",
                            (msg["timestamp"], msg["node_id"], msg["location"], level,
                             label, round(conf, 3), msg["true_state"], exposure, anomaly))
            self.db.commit()
            if exposure:
                self._alert(msg, f"NOISE LIMIT EXCEEDED: {level} dB(A) >= {config.NOISE_LIMIT_DBA}")
            if anomaly:
                self._alert(msg, f"ABNORMAL SOUND: {label} (confidence {conf:.0%})")

    def _alert(self, msg, text):
        line = f"[{msg['timestamp']}] {msg['node_id']} ({msg['location']}): {text}"
        self.alerts.append(line)
        print("ALERT  ", line)

    def stop(self):
        self.running = False
