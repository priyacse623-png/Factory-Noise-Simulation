"""Run the complete simulation: virtual nodes -> transport -> CNN server -> database.

Examples
  python run_simulation.py                       # 60 s, 3 nodes, random faults
  python run_simulation.py --duration 120 --speed 4
  python run_simulation.py --force-state "Bearing fault"   # every node faulty
  python run_simulation.py --transport mqtt      # needs a Mosquitto broker
"""
import argparse, time, sqlite3
import pandas as pd
import config
from src.transport import make_transport
from src.node_simulator import NoiseNode
from src.inference_server import InferenceServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=60, help="simulated seconds per node")
    ap.add_argument("--speed", type=float, default=2.0, help="1 = real time, 4 = 4x faster")
    ap.add_argument("--transport", choices=["queue", "mqtt"], default="queue")
    ap.add_argument("--fault-prob", type=float, default=0.12)
    ap.add_argument("--force-state", choices=config.CLASSES, default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    tr = make_transport(a.transport)
    server = InferenceServer(tr)
    nodes = [NoiseNode(nid, loc, tr, a.duration, a.speed, a.fault_prob, a.force_state, a.seed + i)
             for i, (nid, loc) in enumerate(config.NODES.items())]
    server.start()
    for n in nodes: n.start()
    for n in nodes: n.join()
    t0 = time.time()                   # let the server finish processing queued messages
    while time.time() - t0 < 180 and not tr.q.empty():
        time.sleep(0.5)
    time.sleep(1.5)
    server.stop()

    df = pd.read_sql("SELECT * FROM readings", sqlite3.connect(config.DB_PATH))
    df.to_csv(f"{config.DATA_DIR}/simulation_results.csv", index=False)
    acc = (df.predicted == df.true_state).mean() * 100
    print("\n========== SUMMARY ==========")
    print(f"Readings processed   : {len(df)}")
    print(f"Average level        : {df.level_dba.mean():.1f} dB(A)   (max {df.level_dba.max():.1f})")
    print(f"Exposure alerts      : {int(df.exposure_alert.sum())}")
    print(f"Anomaly alerts       : {int(df.anomaly_alert.sum())}")
    print(f"CNN accuracy vs truth: {acc:.1f} %")
    print(f"Saved: {config.DATA_DIR}/simulation_results.csv  and  {config.DB_PATH}")


if __name__ == "__main__":
    main()
