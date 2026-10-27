"""Message transport between simulated IoT nodes and the server.

Default  : in-process queue (no broker needed -> simplest simulation).
Optional : real MQTT broker (e.g. Mosquitto) using paho-mqtt.
"""
import json, queue
import config


class QueueTransport:
    def __init__(self):
        self.q = queue.Queue()

    def publish(self, msg: dict):
        self.q.put(msg)

    def get(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


class MqttTransport:
    def __init__(self):
        import paho.mqtt.client as mqtt
        self.q = queue.Queue()
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:          # paho-mqtt < 2.0
            self.client = mqtt.Client()
        self.client.on_message = lambda c, u, m: self.q.put(json.loads(m.payload))
        self.client.connect(config.MQTT_HOST, config.MQTT_PORT)
        self.client.subscribe(config.MQTT_TOPIC)
        self.client.loop_start()

    def publish(self, msg: dict):
        self.client.publish(config.MQTT_TOPIC, json.dumps(msg))

    def get(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


def make_transport(kind="queue"):
    return MqttTransport() if kind == "mqtt" else QueueTransport()
