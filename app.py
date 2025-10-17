import streamlit as st
import tensorflow as tf
import numpy as np
import csv
import json
import os
import gspread
from datetime import datetime
from pathlib import Path
from PIL import Image

st.title("Stock Trend Prediction")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Define the model architecture
@st.cache_resource
def my_lenet(do_freq=0.3):
    inputs = tf.keras.layers.Input(shape=(128,128,3))

    c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
    s2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c1)
    s2 = tf.keras.layers.Dropout(do_freq)(s2)

    c3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(s2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(c3)
    s4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c3)
    s4 = tf.keras.layers.Dropout(do_freq)(s4)

    c5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(s4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(c5)
    s6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(c5)
    s6 = tf.keras.layers.Dropout(do_freq)(s6)

    flat = tf.keras.layers.Flatten()(s6)
    f7 = tf.keras.layers.Dense(256, activation='relu')(flat)
    f7 = tf.keras.layers.BatchNormalization()(f7)
    f7 = tf.keras.layers.Dropout(do_freq)(f7)
    f8 = tf.keras.layers.Dense(128, activation='relu')(f7)
    f8 = tf.keras.layers.BatchNormalization()(f8)
    f8 = tf.keras.layers.Dropout(do_freq)(f8)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(f8)

    return tf.keras.models.Model(inputs, outputs, name='my_lenet')

# Load the model once
model = my_lenet()
model.load_weights("best_model.weights.h5")

if uploaded_file is not None:
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_image = original_image.resize((128, 128))
    
        # Show original and resized side by side
        col1, col2 = st.columns(2)
        col1.image(original_image, caption="Original image", width="stretch")
        col2.image(resized_image, caption="Resized (128x128)", width="stretch")
    
        img_array = np.array(resized_image) / 255.0  # Normalize pixel values
        img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension

    try:
        preds = model.predict(img_batch)
        pred_class = int(np.argmax(preds, axis=1)[0])
        pred_class = "Up" if pred_class == 1 else "Down"
        st.write("Predicted class:", pred_class)
    except Exception as e:
        st.warning(f"Model could not be loaded or prediction failed: {e}")
        preds = None

    if preds is not None:

        save_dir = Path("saved_images")
        save_dir.mkdir(parents=True, exist_ok=True)

        # sanitize filename and construct paths
        original_fname = Path(uploaded_file.name).name if hasattr(uploaded_file, "name") else f"original_{datetime.utcnow().timestamp()}.png"
        orig_path = save_dir / original_fname
        resized_path = save_dir / f"resized_{original_fname}"

        try:
            # Save images locally
            original_image.save(orig_path)
            resized_image.save(resized_path)
        except Exception as e:
            st.warning(f"Failed to save images: {e}")

        # Voting form for 5 future periods
        with st.form("vote_form"):
            vote = st.radio("Period 5 (five periods in the future): Was the prediction correct?", ("Yes", "No"), key="period_5")
            submitted = st.form_submit_button("Submit vote")
        votes = {"period_5": vote}

        if submitted:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "prediction": pred_class,
                "original_image_path": str(orig_path),
                "resized_image_path": str(resized_path),
                "votes": votes
            }

            # Try to save votes to Google Sheets if service account creds provided, otherwise fall back to local disk
            GS_CREDS = os.environ.get("GS_CREDENTIALS_JSON")  # full service account JSON as env var
            GS_SHEET_ID = os.environ.get("GS_SHEET_ID")      # Google Sheet ID (from URL)
            GS_SHEET_NAME = os.environ.get("GS_SHEET_NAME", "Sheet1")

            if GS_CREDS and GS_SHEET_ID:
                try:

                    creds_dict = json.loads(GS_CREDS)
                    client = gspread.service_account_from_dict(creds_dict)

                    sh = client.open_by_key(GS_SHEET_ID)
                    try:
                        worksheet = sh.worksheet(GS_SHEET_NAME)
                    except Exception:
                        worksheet = sh.sheet1

                    # ensure header exists
                    values = worksheet.get_all_values()
                    header = ["timestamp", "prediction", "original_image_path", "resized_image_path", "votes_json"]
                    if not values or values[0] != header:
                        try:
                            worksheet.insert_row(header, index=1)
                        except Exception:
                            worksheet.update("A1", [header])

                    row = [
                        record["timestamp"],
                        record["prediction"],
                        record["original_image_path"],
                        record["resized_image_path"],
                        json.dumps(record["votes"]),
                    ]
                    worksheet.append_row(row, value_input_option="USER_ENTERED")
                    st.success("Saved votes to Google Sheet.")
                except Exception as e:
                    st.warning(f"Google Sheets save failed: {e}. Falling back to local save.")
                    try:
                        csv_file = save_dir / "votes.csv"
                        write_header = not csv_file.exists()
                        with open(csv_file, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if write_header:
                                writer.writerow(["timestamp", "prediction", "original_image_path", "resized_image_path", "votes_json"])
                            writer.writerow([record["timestamp"], record["prediction"], record["original_image_path"], record["resized_image_path"], json.dumps(record["votes"])])
                        st.success("Votes saved locally. Note: local storage may be ephemeral in cloud environments.")
                    except Exception as e_local:
                        st.error(f"Failed to save locally as fallback: {e_local}")
            else:
                # No Google Sheets config found -> save locally (may be ephemeral in cloud)
                csv_file = save_dir / "votes.csv"
                write_header = not csv_file.exists()

                try:
                    with open(csv_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(["timestamp", "prediction", "original_image_path", "resized_image_path", "votes_json"])
                        writer.writerow([record["timestamp"], record["prediction"], record["original_image_path"], record["resized_image_path"], json.dumps(record["votes"])])
                    st.success("Votes saved locally. Note: local storage may be ephemeral in cloud environments.")
                except Exception as e:
                    st.error(f"Failed to save votes locally: {e}")
