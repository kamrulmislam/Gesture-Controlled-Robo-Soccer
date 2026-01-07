import os
# import openvino as ov
from scipy.signal import windows, find_peaks
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

def generate_range_doppler_profiles_per_antenna(data: np.ndarray) -> np.ndarray:
    """
    Compute the range-Doppler profile for each (n_frame, n_antenna, n_chirp, n_sample) matrix.

    Args:
        data (np.ndarray): Input matrix for of size
                           (n_frame, n_antenna, n_chirp, n_sample).

    Returns:
        np.ndarray: Range-Doppler profile with shape (n_frame,n_antenna, n_chirp, n_sample // 2 + 1).
    """
    # Extract the number of frames, antennas, chirps, and samples
    n_frame, n_antenna, n_chirp, n_sample = data.shape

    # Step 1: Remove the DC bias (mean) along the sample dimension
    data_centered = data - np.mean(data, axis=-1, keepdims=True)

    # Step 2: Apply the Blackman-Harris window to reduce spectral leakage
    range_window = windows.blackmanharris(n_sample).reshape(1, n_sample) # Shape for broadcasting
    data_windowed = data_centered * range_window

    # Step 3: Perform FFT along the sample axis and normalize by the window sum
    range_fft = np.fft.fft(data_windowed, axis=-1) / np.sum(range_window)

    # Step 4: Remove quasi-static targets by subtracting the mean over chirps
    range_fft -= np.mean(range_fft, axis=2, keepdims=True)

    # Step 5: Retain only the positive spectrum
    range_fft_half = range_fft[..., :n_sample // 2 + 1]
    range_fft_half[:,:,:,1::-1] = 2*range_fft_half[:,:,:,1::-1]

    # Step 6: Apply the Blackman-Harris window to the chirp axis
    doppler_window = windows.blackmanharris(n_chirp).reshape(1, 1, n_chirp, 1)
    range_fft_half_windowed = range_fft_half * doppler_window


    # Step 7: Perform FFT on the chirp axis and apply FFT shift for Doppler centering
    range_doppler = np.fft.fftshift(np.fft.fft(range_fft_half_windowed, axis=2), axes=2)/np.sum(doppler_window)

    return range_doppler

def generate_range_doppler_profiles(data: np.ndarray) -> np.ndarray:
    """
    Computes the summed range-Doppler profile over antennas.

    Args:
        data (np.ndarray): Input matrix of shape
                           (n_frame, n_antenna, n_chirp, n_sample).

    Returns:
        np.ndarray: Range-Doppler profile with shape (n_frame, n_chirp, n_sample // 2 + 1).
    """
    # Validate input dimensions
    if data.ndim != 4:
        raise ValueError("Input data must be a 4D array with shape (n_frame, n_antenna, n_chirp, n_sample).")

    n_frame, n_antenna, n_chirp, n_sample = data.shape

    # Compute range-Doppler profiles for each antenna
    range_doppler = generate_range_doppler_profiles_per_antenna(data)

    # Integrate the absolute FFT data across antennas
    integrated_range_doppler = np.sum(np.abs(range_doppler), axis=1) / n_antenna

    return integrated_range_doppler

def run_inference_on_raw_data(raw_data, model):
    """
    Inference function for raw radar data.
    """
    # calculate doppler
    range_doppler_profile = generate_range_doppler_profiles(raw_data)
    if range_doppler_profile.ndim == 3:
        range_doppler_profile = np.expand_dims(range_doppler_profile, axis=-1)
    # print(range_doppler_profile.shape)

    X_test_1 = np.expand_dims(range_doppler_profile, axis=0)
    # print(X_test_1.shape)

    # # Generate predictions
    output_layer = model.output(0)
    y_test_predictions_prob = model([X_test_1])[output_layer]
    y_test_prediction = np.argmax(y_test_predictions_prob, axis=1)

    return y_test_prediction

def load_openvino_model(model_path: str, device: str = "CPU"):
    """
    Load an OpenVINO model from the specified path.

    Args:
        model_path (str): Path to the OpenVINO model (.xml file).
        device (str): Device to load the model on (default is "CPU").

    Returns:
        ov.CompiledModel: The compiled OpenVINO model.
    """
    core = ov.Core()
    model = core.read_model(model=model_path)
    print(device)
    compiled_model = core.compile_model(model, device)
    return compiled_model

def get_device():
    """
    Get the available device (GPU if available, else CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return device

def get_transform(IMG_SIZE):
    """
    Get the image transformation pipeline for testing.

    Args:
        IMG_SIZE (int): Desired image size (height and width) after resizing.

    Returns:
        torchvision.transforms.Compose: Composed transformations for test images.
    """
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return test_transforms

def build_convnext_tiny(num_classes, weights='DEFAULT', freeze_layers=True):
    """
    Builds and modifies a pre-trained ConvNeXt Tiny model.
    """
    # Load Pre-trained ConvNeXt Tiny
    # 'DEFAULT' weights are usually ImageNet-1K
    model = models.convnext_tiny(weights=weights)

    if freeze_layers:
        # Freeze all parameters in the feature extraction layers
        for param in model.parameters():
            param.requires_grad = False

    # Replace the Classifier Head
    # ConvNeXt's classifier is usually a Sequential block.
    # We replace the last linear layer.
    last_layer_input = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(last_layer_input, num_classes)

    return model

def build_resnet18(num_classes, weights='IMAGENET1K_V1', freeze_layers=True):
    """
    Builds and modifies a pre-trained ResNet18 model.
    """
    # Load ResNet18 pre-trained on ImageNet
    model = models.resnet18(weights=weights)

    if freeze_layers:
        # Freeze all parameters in the feature extraction layers
        for param in model.parameters():
            param.requires_grad = False

    # Get the number of input features to the final fully-connected (fc) layer
    num_ftrs = model.fc.in_features

    # Replace the classification head to match the number of classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    print(f"ResNet18 loaded. Final layer output set to {num_classes} classes.")
    return model

def build_efficientnet_b0(num_classes, weights='DEFAULT', freeze_layers=True):
    """
    Builds and modifies a pre-trained EfficientNet-B0 model.
    """
    # Load EfficientNet-B0 pre-trained on ImageNet
    # 'DEFAULT' uses IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    if freeze_layers:
        # Freeze all parameters in the feature extraction layers
        for param in model.features.parameters():
            param.requires_grad = False

    # EfficientNet's classifier is a Sequential layer. We replace the last Linear layer (index [1])
    # The input features are taken from the second to last layer
    last_layer_input = model.classifier[1].in_features

    # Create a new classifier with the correct output size
    model.classifier[1] = nn.Linear(last_layer_input, num_classes)

    print(f"EfficientNet-B0 loaded. Final layer output set to {num_classes} classes.")
    return model

def build_mobilenet_v3_large(num_classes, weights='DEFAULT', freeze_layers=True):
    """
    Builds MobileNetV3 Large.
    """
    model = models.mobilenet_v3_large(weights=weights)
    if freeze_layers:
        for param in model.features.parameters():
            param.requires_grad = False

    # MobileNetV3 classifier is a Sequential block. The last layer is the Linear one.
    # model.classifier[-1] is the final Linear layer
    last_layer_input = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(last_layer_input, num_classes)

    print(f"MobileNetV3 Large loaded. Final layer output set to {num_classes} classes.")
    return model

def build_densenet121(num_classes, weights='DEFAULT', freeze_layers=True):
    """
    Builds DenseNet121.
    """
    model = models.densenet121(weights=weights)
    if freeze_layers:
        for param in model.features.parameters():
            param.requires_grad = False

    # DenseNet classifier is a single Linear layer named 'classifier'
    last_layer_input = model.classifier.in_features
    model.classifier = nn.Linear(last_layer_input, num_classes)

    print(f"DenseNet121 loaded. Final layer output set to {num_classes} classes.")
    return model

def build_resnet50(num_classes, weights='DEFAULT', freeze_layers=True):
    """
    Builds ResNet50.
    """
    model = models.resnet50(weights=weights)
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    print(f"ResNet50 loaded. Final layer output set to {num_classes} classes.")
    return model

def build_vgg16(num_classes, weights='DEFAULT', freeze_layers=True):
    """
    Builds VGG16 (with Batch Normalization).
    """
    # Using BN version for better training stability
    model = models.vgg16_bn(weights=weights)

    if freeze_layers:
        for param in model.features.parameters():
            param.requires_grad = False

    # VGG classifier is a Sequential block.
    # The last layer is at index 6
    last_layer_input = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(last_layer_input, num_classes)

    print(f"VGG16 (BN) loaded. Final layer output set to {num_classes} classes.")
    return model

def build_vgg19(num_classes, weights='DEFAULT', freeze_layers=True):
    """
    Builds VGG19 (with Batch Normalization).
    """
    model = models.vgg19_bn(weights=weights)
    if freeze_layers:
        for param in model.features.parameters():
            param.requires_grad = False

    last_layer_input = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(last_layer_input, num_classes)

    print(f"VGG19 (BN) loaded. Final layer output set to {num_classes} classes.")
    return model

def get_model(model_name, NUM_CLASSES):
    """
    Returns the model based on the model name.
    """
    if model_name == 'convnext_tiny':
        return build_convnext_tiny(NUM_CLASSES, None)

    elif model_name == 'resnet18':
        return build_resnet18(NUM_CLASSES, None)

    elif model_name == 'efficientnet_b0':
        return build_efficientnet_b0(NUM_CLASSES, None)

    elif model_name == 'mobilenet_v3_large':
        return build_mobilenet_v3_large(NUM_CLASSES, None)

    elif model_name == 'densenet121':
        return build_densenet121(NUM_CLASSES, None)

    elif model_name == 'resnet50':
        return build_resnet50(NUM_CLASSES, None)

    elif model_name == 'vgg16':
        return build_vgg16(NUM_CLASSES, None)

    elif model_name == 'vgg19':
        return build_vgg19(NUM_CLASSES, None)

    else:
        raise ValueError(f"Model {model_name} not implemented.")

def load_model(model_path, num_classes):
    """
    Load the model and its weights from the specified path.
    """
    device = get_device()
    model_name = model_path.split('/')[-1].split('.')[-2].split('_model_')[1]
    model = get_model(model_name, num_classes)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Weights loaded successfully.")
    else:
        print(f"Error: '{model_path}' not found. Train the model first.")
        return None
    
    model = model.to(device)
    return model, device

def predict_single_image(model, image_path, transform, CLASSES, device):
    """
    Reads an image, applies transforms, and predicts the class.
    """
    model.eval()

    # Load Image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Apply Transform (Must match Validation/Test transforms)
    # Add batch dimension (C, H, W) -> (1, C, H, W)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

        # Get Probabilities using Softmax
        probs = F.softmax(outputs, dim=1)

        # Get Top Prediction
        max_prob, pred_idx = torch.max(probs, 1)

    predicted_class = CLASSES[pred_idx.item()]
    confidence = max_prob.item() * 100

    return predicted_class, confidence, probs[0].cpu().numpy()