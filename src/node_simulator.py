"""Virtual ESP32 noise node.

Each node 'listens' to a machine whose state changes randomly over time
(mostly Normal, occasionally a fault). Every second it measures dB(A) and
publishes the audio window (as a float list) like the real node would send
its features over MQTT.
"""
import time, threading, datetime
import numpy as np
import config
from src.signal_generator import generate
from src.features import spl_dba


class NoiseNode(threading.Thread):
    def __init__(self, node_id, location, transport, duration=60, speed=1.0,
                 fault_prob=0.12, forced_state=None, seed=None):
        super().__init__(daemon=True)
        self.node_id, self.location = node_id, location
        self.transport, self.duration, self.speed = transport, duration, speed
        self.fault_prob, self.forced_state = fault_prob, forced_state
        self.rng = np.random.default_rng(seed)
        self.state, self.state_left = config.NORMAL_CLASS, 0

    def _next_state(self):
        if self.forced_state:
            return self.forced_state
        if self.state_left > 0:
            self.state_left -= 1
            return self.state
        if self.rng.random() < self.fault_prob:
            self.state = str(self.rng.choice(config.CLASSES[1:]))
            self.state_left = int(self.rng.integers(3, 8))       # fault lasts 3-8 s
        else:
            self.state = config.NORMAL_CLASS
        return self.state

    def run(self):
        for _ in range(int(self.duration)):
            truth = self._next_state()
            x = generate(truth, self.rng)
            self.transport.publish({
                "node_id": self.node_id,
                "location": self.location,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "level_dba": round(spl_dba(x), 1),    # measured on the 'ESP32'
                "true_state": truth,                  # ground truth (for evaluation only)
                "audio": x.astype(np.float32).tolist(),
            })
            time.sleep(1.0 / self.speed)
