import numpy as np
import torch
import torch.nn as nn

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        # Convolutional Feature Extractor with BatchNorm
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32 x 16 x 16
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 8 x 8
            
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128 x 4 x 4
        )
        
        # Fully Connected Classifier with Dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128 * 4 * 4, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def fuse_batch_layer(conv_weights, conv_bias, bn_gamma, bn_beta, bn_mean, bn_var, epsilon=1e-5):
    """
    Fuses the parameters of a Conv2D and a BatchNorm2D layer into a single
    set of fused weights and biases for a single convergence layer.

    Args:
        conv_weights (np.ndarray): The weights from the Conv2D layer. 
                                   Shape: (out_channels, in_channels, kernel_h, kernel_w)
        conv_bias (np.ndarray): The bias from the Conv2D layer.
                                Shape: (out_channels,)
        bn_gamma (np.ndarray): The learned 'weight' parameter from the BatchNorm2D layer.
                               Shape: (out_channels,)
        bn_beta (np.ndarray): The learned 'bias' parameter from the BatchNorm2D layer.
                              Shape: (out_channels,)
        bn_mean (np.ndarray): The running_mean buffer from the BatchNorm2D layer.
                              Shape: (out_channels,)
        bn_var (np.ndarray): The running_var buffer from the BatchNorm2D layer.
                             Shape: (out_channels,)
        epsilon (float): The epsilon value from the BatchNorm2D layer.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - fused_weights (np.ndarray): The new weights for the combined layer.
            - fused_bias (np.ndarray): The new bias for the combined layer.
    """
    # 1. Calculate the standard deviation
    std_dev = np.sqrt(bn_var + epsilon)
    
    # 2. Compute the scaling factor
    scale_factor = bn_gamma / std_dev
    
    # 3. Compute the fused bias
    # Formula: fused_bias = gamma * (conv_bias - mean) / std + beta
    fused_bias = scale_factor * (conv_bias - bn_mean) + bn_beta

    # 4. Compute the fused weights
    # Formula: fused_weights = gamma * conv_weights / std
    # We need to reshape the scale_factor for broadcasting
    # from (out_channels,) to (out_channels, 1, 1, 1)
    scale_factor_reshaped = scale_factor[:, np.newaxis, np.newaxis, np.newaxis]
    fused_weights = scale_factor_reshaped * conv_weights
    
    return fused_weights, fused_bias

def save_array_to_coe(data_array, filename="data.coe", radix=16):
    """
    Saves a 1-D NumPy array of 4-byte elements to a Xilinx/AMD .coe file.

    Args:
        data_array (np.ndarray): The 1-D array with 3072 elements (e.g., dtype=np.int32 or np.float32).
        filename (str): The name of the output .coe file.
        radix (int): The number base for the data (2, 10, or 16). 16 (hex) is standard.
    """

    # 2. Ensure the element size is 4 bytes (e.g., 'int32', 'float32')
    if data_array.dtype.itemsize != 4:
        raise ValueError("Array elements must be 4 bytes (32 bits) each.")

    # 3. Convert the array to a view of unsigned 32-bit integers.
    data_32bit = data_array.view(np.uint32)

    # 4. Determine the data width based on 4 bytes = 32 bits
    data_width = 32

    # 5. Format the header and data
    header = f"memory_initialization_radix={radix};\nmemory_initialization_vector=\n"

    # Format the data based on the chosen radix
    if radix == 16:
        # Format as 8-character hex strings (32 bits = 8 hex nibbles)
        data_lines = [f"{x:08X}" for x in data_32bit]
    elif radix == 10:
        data_lines = [str(x) for x in data_32bit]
    elif radix == 2:
        # Format as 32-character binary strings
        data_lines = [f"{x:032b}" for x in data_32bit]
    else:
        raise ValueError("Radix must be 2, 10, or 16.")

    # 6. Join the data with a comma and then add a semicolon at the end
    data_string = ',\n'.join(data_lines) + ';'

    # 7. Write to the .coe file
    try:
        with open(filename, 'w') as f:
            f.write(header)
            f.write(data_string)
        print(f"Successfully saved {data_array.size} words (32-bit each) to {filename}")
    except IOError as e:
        print(f"Error writing to file: {e}")






MODEL_PATH = "cifar10_improved_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImprovedCNN().to(device)
    
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except (FileNotFoundError, RuntimeError) as e:
    print(f"Error loading model: {e}")
    exit()


conv1_w2 = model.state_dict()["features.0.weight"].cpu().numpy()
conv1_b2 = model.state_dict()["features.0.bias"].cpu().numpy()
bn1_w2 = model.state_dict()["features.1.weight"].cpu().numpy()
bn1_b2 = model.state_dict()["features.1.bias"].cpu().numpy()
bn1_mean = model.state_dict()["features.1.running_mean"].cpu().numpy() 
bn1_var = model.state_dict()["features.1.running_var"].cpu().numpy()

b1_w, b1_b = fuse_batch_layer(conv1_w2, conv1_b2, bn1_w2, bn1_b2, bn1_mean, bn1_var)

save_array_to_coe(np.append(b1_w, b1_b))