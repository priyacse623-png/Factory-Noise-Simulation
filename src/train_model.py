"""Generate a synthetic dataset, train the CNN and save results.

Run:  python -m src.train_model --per-class 300 --epochs 15
Outputs: models/noise_cnn.keras, data/confusion_matrix.png,
         data/classification_report.txt, data/training_curve.png
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import config
from src.signal_generator import generate
from src.features import log_mel
from src.model import build_model


def make_dataset(per_class, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for idx, label in enumerate(config.CLASSES):
        for _ in range(per_class):
            X.append(log_mel(generate(label, rng)))
            y.append(idx)
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    print("Generating synthetic dataset ...")
    X, y = make_dataset(args.per_class)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=1)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=1)
    print(f"train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    model = build_model()
    model.summary()
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=args.epochs, batch_size=32, verbose=2)
    model.save(config.MODEL_PATH)

    pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    rep = classification_report(y_te, pred, target_names=config.CLASSES, digits=3)
    print(rep)
    with open(os.path.join(config.DATA_DIR, "classification_report.txt"), "w") as f:
        f.write(rep)

    cm = confusion_matrix(y_te, pred)
    ConfusionMatrixDisplay(cm, display_labels=config.CLASSES).plot(cmap="Blues", xticks_rotation=30)
    plt.title("Confusion matrix (test set)"); plt.tight_layout()
    plt.savefig(os.path.join(config.DATA_DIR, "confusion_matrix.png"), dpi=150); plt.close()

    plt.plot(hist.history["accuracy"], label="train"); plt.plot(hist.history["val_accuracy"], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Training curve"); plt.tight_layout()
    plt.savefig(os.path.join(config.DATA_DIR, "training_curve.png"), dpi=150); plt.close()
    print("Model saved to", config.MODEL_PATH)


if __name__ == "__main__":
    main()
