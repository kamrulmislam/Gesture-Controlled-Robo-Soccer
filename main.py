import tkinter as tk
import time
import json
import pprint
import numpy as np
from InfineoenManager import InfineonManager as RadarManager
from prediction_utils import load_openvino_model, run_inference_on_raw_data, load_model, get_transform
from worker import record_and_predict, record_and_predict_keras
from vex import *
from vex.vex_globals import *
from vex_control import send_command_to_vex
from prediction_utils_tf import load_keras_model, load_labels

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

## Record Configuration
n_frames = 20

chirp_config_path = 'configs/cfg_simo_chirp.json'
seq_config_path = 'configs/cfg_simo_seq.json'

# chirp_config_path = 'configs/cfg_chirp.json'
# seq_config_path = 'configs/cfg_seq.json'

with open(chirp_config_path) as file:
    cfg_chirp = json.load(file)

with open(seq_config_path) as file:
    cfg_seq = json.load(file)

radar = RadarManager()

## Calculate Radar Parameters
params = radar.get_params({**cfg_chirp, **cfg_seq})
# print('Radar Parameters:')
# pprint.pprint(params)

## Prediction Configuration
# Define a dictionary mapping gestures to numeric labels
gesture_dict = {
    'SwipeLeft': 0,
    'SwipeRight': 1,
    'SwipeDown': 2,
    'SwipeUp': 3,
    'Push': 4
}
CLASSES = ["up", "down", "right", "left", "push"]
IMG_SIZE = 224

### OpenVino for Lunar Lake
# model_path = './rd_model.xml'
# model = load_openvino_model(model_path, device="GPU") # device = 'CPU'or 'GPU'

### PyTorch Model
# model_path = './saved_model/best_model_vgg19.pth'
# model_info = load_model(model_path, len(CLASSES))
# if model_info is not None:
#     model, device = model_info
#     print('Model loaded successfully.')
# else:
#     print('Model could not be loaded.')

# test_transforms = get_transform(IMG_SIZE=IMG_SIZE)

### Keras Model
model_path = '/media/ci4r/Extra/Kamrul_files/Robo-Soccer/saved_model/converted_keras/keras_model.h5'
model = load_keras_model(model_path)
labels_path = '/media/ci4r/Extra/Kamrul_files/Robo-Soccer/saved_model/converted_keras/labels.txt'
class_names = load_labels(labels_path)

## Vex Configuration
distance_unit_in_mm = 50

# Robot initialization for AIM platform
robot = Robot()

## Task
def start_recording():
    # Update the UI to show the task has started
    status_label.config(text="Status: Recording...")
    record_button.config(state="disabled", text="Busy...")
    
    # Force the GUI to update immediately so you see the text change
    root.update() 

    # Task
    print("Task started...")

    # prediction = record_and_predict(radar, n_frames, cfg_chirp, cfg_seq, params, model, test_transforms, CLASSES, device) # For PyTorch/OpenVino
    prediction = record_and_predict_keras(radar, n_frames, cfg_chirp, cfg_seq, params, model, class_names) # For Keras/TensorFlow

    response = send_command_to_vex(robot, prediction.lower(), distance_unit_in_mm)
    if response["status"] == "success":
        status_label.config(text=f"Status: Robot Moved...")
    else:
        status_label.config(text=f"Status: Robot Failed to Move...")
    print("Task finished.")

    status_label.config(text="Status: Ready")
    record_button.config(state="normal", text="Record")

## GUI Setup
# Main window
root = tk.Tk()
root.title("Radar Gesture Recorder")
root.geometry("300x150")

# Show status
status_label = tk.Label(root, text="Status: Ready", font=("Arial", 10))
status_label.pack(pady=20)

# Record button
record_button = tk.Button(root, text="Record", command=start_recording, height=2, width=10)
record_button.pack()

# GUI event loop (this handles the "waiting")
root.mainloop()