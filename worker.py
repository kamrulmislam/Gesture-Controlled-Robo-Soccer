import time
import numpy as np
import datetime
import os
from processing_utils import spectrogram
from prediction_utils import run_inference_on_raw_data, predict_single_image
from prediction_utils_tf import preprocess_image, predict_image

def record_and_predict(radar, n_frames, cfg_chirp, cfg_seq, params, model, test_transforms, CLASSES, device):
    """
    Record radar data, generate spectrogram, and predict gesture class.
    """

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_file_name = f"./data/test/raw_data/{timestamp}_raw.npy"
    md_file_name = f"./data/test/spectrogram/{timestamp}_spect.png"

    ## Record
    start_time = time.time()
    radar.init_device_fmcw(cfg_seq, cfg_chirp)
    data = radar.fetch_n_frames(n_frames)
    radar.close()
    print(f"Collected data in {time.time() - start_time:.2f} seconds")
    print('data shape:', data.shape)  # (n_frame, n_antenna, n_chirp, n_sample)
    np.save(raw_file_name, data)

    ## Spectrogram
    spectrogram(data, duration=data.shape[0]*cfg_seq['frame_repetition_time_s'], prf=params['prf'], mti=True, is_save=True, savename=md_file_name)
    # spectrogram(data, duration=data.shape[0]*cfg_seq['frame_repetition_time_s'], prf=params['prf'], mti=True)

    ## Predict
    start_time = time.time()
    # prediction = run_inference_on_raw_data(data, model)
    if os.path.exists(md_file_name):
        pred_class, conf, all_probs = predict_single_image(model, md_file_name, test_transforms, CLASSES, device)

        print(f"Prediction: {pred_class.upper()}")
        print(f"Confidence: {conf:.2f}%")
        print("Class Probabilities:")
        for i, cls in enumerate(CLASSES):
            print(f"  {cls}: {all_probs[i]*100:.2f}%")

    else:
        print(f"Test image not found: {md_file_name}")

    # print(f"Predicted {pred_class} in {time.time() - start_time:.2f} seconds")
    
    return pred_class

def record_and_predict_keras(radar, n_frames, cfg_chirp, cfg_seq, params, model, class_names):
    """
    Record radar data, generate spectrogram, and predict gesture class using Keras model.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_file_name = f"./data/test/raw_data/{timestamp}_raw.npy"
    md_file_name = f"./data/test/spectrogram/{timestamp}_spect.png"

    ## Record
    start_time = time.time()
    radar.init_device_fmcw(cfg_seq, cfg_chirp)
    data = radar.fetch_n_frames(n_frames)
    radar.close()
    print(f"Collected data in {time.time() - start_time:.2f} seconds")
    print('data shape:', data.shape)  # (n_frame, n_antenna, n_chirp, n_sample)
    np.save(raw_file_name, data)

    ## Spectrogram
    spectrogram(data, duration=data.shape[0]*cfg_seq['frame_repetition_time_s'], prf=params['prf'], mti=True, is_save=True, savename=md_file_name)
    # spectrogram(data, duration=data.shape[0]*cfg_seq['frame_repetition_time_s'], prf=params['prf'], mti=True)

    ## Predict
    start_time = time.time()
    # prediction = run_inference_on_raw_data(data, model)
    if os.path.exists(md_file_name):
        index, confidence_score = predict_image(model, preprocess_image(md_file_name))

        class_name = class_names[index][2:].strip()

        print(f"Prediction: {class_name}")
        # print(f"Confidence: {confidence_score*100:.2f}%")
        # print(f"Confidence: {confidence_score}")
        for i, cls in enumerate(class_names):
            print(f"  {cls.strip()}: {confidence_score[i]*100:.2f}%")

    else:
        print(f"Test image not found: {md_file_name}")

    # print(f"Predicted {pred_class} in {time.time() - start_time:.2f} seconds")
    
    return class_name