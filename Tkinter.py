import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import random

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\Destiny\Downloads\Intel Classifier\intel_model.h5")
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

IMG_SIZE = (224, 224)
PRED_FOLDER = r"C:\Users\Destiny\Downloads\Intel Classifier\archive\seg_pred\seg_pred"
CONFUSION_MATRIX_PATH = r"C:\Users\Destiny\Downloads\Intel Classifier\ConfusionMatrix.png"
RESULT_PLOT_PATH = r"C:\Users\Destiny\Downloads\Intel Classifier\Results.png"
BEST_ACCURACY = "91.63%"

def load_and_prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def display_and_classify(img_path):
    # Show image
    img = Image.open(img_path)
    img_resized = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img_resized)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    # Predict
    processed_image = load_and_prepare_image(img_path)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    result_label.config(text=f"Prediction: {predicted_class}", fg="blue")

def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        display_and_classify(file_path)

def classify_random_image():
    all_images = [os.path.join(PRED_FOLDER, f) for f in os.listdir(PRED_FOLDER)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if all_images:
        random_path = random.choice(all_images)
        display_and_classify(random_path)

def show_image(path, label_widget):
    img = Image.open(path)
    img_resized = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_resized)
    label_widget.configure(image=img_tk)
    label_widget.image = img_tk

def show_confusion_matrix():
    show_image(CONFUSION_MATRIX_PATH, image_label)
    result_label.config(text="Confusion Matrix")

def show_results_plot():
    show_image(RESULT_PLOT_PATH, image_label)
    result_label.config(text=f"Best Accuracy: {BEST_ACCURACY}", fg="green")

# GUI Setup
root = tk.Tk()
root.title("Intel Image Classifier")
root.geometry("450x600")

btn_choose = tk.Button(root, text="Choose Image", command=classify_image)
btn_choose.pack(pady=10)

btn_random = tk.Button(root, text="Classify Random Image", command=classify_random_image)
btn_random.pack(pady=10)

btn_conf_matrix = tk.Button(root, text="Confusion Matrix", command=show_confusion_matrix)
btn_conf_matrix.pack(pady=10)

btn_results = tk.Button(root, text="Results", command=show_results_plot)
btn_results.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
