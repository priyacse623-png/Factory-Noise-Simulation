# Factory Noise Monitoring System – Simulation Prototype
IoT + Deep Neural Network (CNN) – fully software simulation, no hardware needed.

## What is simulated
| Real system | In this simulation |
|---|---|
| INMP441 mic + ESP32 node | `src/node_simulator.py` (virtual nodes, synthetic machine sound) |
| dB(A) calculation on ESP32 | `src/features.py` (A-weighting filter, SPL in dB(A)) |
| Wi-Fi + MQTT broker | `src/transport.py` (in-memory queue, or real MQTT with `--transport mqtt`) |
| Server + CNN inference | `src/inference_server.py` + `src/model.py` |
| Database | SQLite `logs/noise_log.db` |
| Dashboard / alerts | `dashboard/app.py` (Streamlit) + console alerts |

## File structure
```
factory_noise_sim/
├── README.md
├── requirements.txt
├── config.py                  # all settings (limits, classes, nodes, MQTT)
├── run_simulation.py          # MAIN: runs nodes -> transport -> CNN -> DB
├── src/
│   ├── signal_generator.py    # synthetic sounds: Normal, Bearing fault, Air leak, Impact, Overload
│   ├── features.py            # dB(A) + log-mel spectrogram
│   ├── model.py               # CNN architecture (Table 6.2 of the report)
│   ├── train_model.py         # dataset generation + training + evaluation plots
│   ├── transport.py           # queue / MQTT
│   ├── node_simulator.py      # virtual ESP32 noise node
│   └── inference_server.py    # CNN inference + alert rules + database
├── dashboard/
│   └── app.py                 # live Streamlit dashboard
├── models/noise_cnn.keras     # trained model (created by train_model.py)
├── data/                      # confusion_matrix.png, training_curve.png,
│                              # classification_report.txt, simulation_results.csv
└── logs/noise_log.db          # SQLite log (created by run_simulation.py)
```

## How to run (Python 3.10 – 3.12)
```bash
pip install -r requirements.txt

# 1. Train the CNN on synthetic data  (~2-3 min on CPU)
python -m src.train_model --per-class 300 --epochs 15

# 2. Run the simulation (3 virtual nodes, 60 s, 2x speed)
python run_simulation.py
python run_simulation.py --duration 120 --speed 4 --fault-prob 0.25
python run_simulation.py --force-state "Bearing fault"      # force a fault

# 3. Open the dashboard (new terminal; keep refreshing while simulation runs)
streamlit run dashboard/app.py
```
Tip: start the dashboard first, then run the simulation in another terminal to see live updates.

### Optional: real MQTT
Install Mosquitto, start it (`mosquitto`), then run `python run_simulation.py --transport mqtt`.

## Alert rules (config.py)
* Exposure alert: level >= 85 dB(A) (NIOSH limit)
* Anomaly alert: CNN class != Normal and confidence >= 0.80

## Tune / extend
* Change nodes, limits, classes in `config.py`
* New fault type: add a branch in `signal_generator.py` + add name to `CLASSES`, retrain
* Real data: replace `make_dataset()` in `train_model.py` with MIMII / your own recordings (16 kHz, 1 s clips)
* Real hardware later: ESP32 publishes the same JSON fields to the MQTT topic

## Note for report
The sounds are synthetic, so the accuracy (~95-99 %) shows that the pipeline works; it is not a
real-factory accuracy. Mention this honestly in the Result section.
