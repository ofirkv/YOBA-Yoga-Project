import numpy as np
import pandas as pd
from pathlib import Path
import cv2

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers


def cnn_hand():
    print("Starting CNN hand gesture classification...")
    # Base directory = project root (Mini-Project/)
    base_dir = Path(__file__).resolve().parents[1]

    # Paths
    labels_path = base_dir / "data" / "labels_processed.csv"
    images_dir = base_dir / "data" / "processed"

    print("Labels CSV:", labels_path)
    print("Images folder:", images_dir)

    # -------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------
    def preprocess_image(image_path, img_size=128, use_binary=False):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)

        if use_binary:
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = bw
        else:
            img = gray

        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        return img

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    df = pd.read_csv(labels_path)

    print("CSV head:")
    print(df.head())

    label_map = {"lowered": 0, "raised": 1}
    df["label_id"] = df["label"].map(label_map)

    X, y = [], []

    for idx, row in df.iterrows():
        filename = row["filename"]
        label_id = row["label_id"]

        img_path = images_dir / filename

        if not img_path.exists():
            print(f"Warning: missing image {img_path}")
            continue

        try:
            img = preprocess_image(img_path, img_size=128, use_binary=True)
            X.append(img)
            y.append(label_id)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label distribution:", np.bincount(y))

    # -------------------------------------------------------------------
    # Train/val split
    # -------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train shapes:", X_train.shape, y_train.shape)
    print("Val shapes:  ", X_val.shape, y_val.shape)

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------
    def build_model():
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ])
        
        model = keras.Sequential([
            layers.Input(shape=(128, 128, 1)),

            data_augmentation,

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        model.summary()
        return model

    model = build_model()

    checkpoint_path = base_dir / "best_hand_cnn.weights.h5"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=30,
        callbacks=callbacks,
        shuffle=True,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    return {
        "status": "ok",
        "loss": float(f"{val_loss:.2f}"),
        "acc": float(f"{val_acc:.2f}")
    }

result = cnn_hand()
print(result)