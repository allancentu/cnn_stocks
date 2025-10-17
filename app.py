import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Stock Trend Prediction")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Define the model architecture
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
    original_image = Image.open(uploaded_file).convert("RGB")
    resized_image = original_image.resize((128, 128))
    img_array = np.array(resized_image) / 255.0  # Normalize pixel values
    img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension

    try:
        preds = model.predict(img_batch)
        pred_class_idx = int(np.argmax(preds, axis=1)[0])
        pred_class = "up" if pred_class_idx == 1 else "down"
        st.badge("Image processed successfully", icon=":material/check:", color="green")
        st.write("Based on our model, this asset price will go ", pred_class)
    except Exception as e:
        st.warning(f"Model could not be loaded or prediction failed: {e}")
        preds = None
    
    # Show original and resized side by side
    tab1, tab2 = st.tabs(["Original Image", "Resized Image"])

    with tab1:
        st.image(original_image, caption="Original image", width="content")
    with tab2:
        st.image(resized_image, caption="Resized (128x128)", width="content")
