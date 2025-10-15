import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
import serial
import struct
import time

# --- Configuration ---
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
NUM_FLOATS_TO_SEND = 3072
NUM_FLOATS_TO_RECEIVE = 10

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

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def FPGA_SEND(image):
    """
    Main function to communicate with the FPGA.
    Sends a start command and 3072 floats, then receives 10 floats back.
    """
    
    
    # The data must be sent as bytes. We create a byte array to hold it.
    data_to_send = bytearray()
    
    # 1. Add the start command
    start_command = 0x41
    write_command = 0x42
    data_to_send.append(start_command)
    
    # 2. Pack the floating-point numbers into the byte array
    print(f"Preparing {NUM_FLOATS_TO_SEND} floats for transmission...")
    for f in image:
        packed_float = struct.pack('>f', f)
        data_to_send.extend(packed_float)
        
    print(f"Amount of bytes sent: {len(data_to_send)}")

    ser = None  # Initialize to None
    try:
        # 3. Open the serial port
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=10  # Set a timeout for read operations (in seconds)
        )
        print(f"Serial port {SERIAL_PORT} opened successfully at {BAUD_RATE} baud.")
        
        # 4. Send the data to the FPGA
        print("Sending data...")
        ser.write(data_to_send)
        print("Data sent. Waiting for 1 seconds for computation...")
        time.sleep(1)

        data_to_send = bytearray()
        data_to_send.append(write_command)
        ser.write(data_to_send)
        print("Sending write command...")

        # 5. Receive data back from the FPGA
        bytes_to_receive = NUM_FLOATS_TO_RECEIVE * 4
        received_bytes = ser.read(40)
        
        if len(received_bytes) < bytes_to_receive:
            print("\n--- Error ---")
            print(f"Timeout: Expected {bytes_to_receive} bytes but received only {len(received_bytes)}.")
            print("Please check the FPGA design, connections, and ensure it's running correctly.")
            return

        # 6. Unpack the received bytes into floating-point numbers
        # The VHDL sends the 32-bit words in big-endian byte order
        received_floats = []
        for i in range(NUM_FLOATS_TO_RECEIVE):
            chunk = received_bytes[i*4 : (i+1)*4]
            unpacked_float = struct.unpack('>f', chunk)[0]
            received_floats.append(unpacked_float)
            
        # 7. Print the results
        print("\n--- Success! ---")
        print(f"Received {len(received_floats)} floats from the FPGA:")
        for i, f in enumerate(received_floats):
            print(f"  Float {i+1}: {f}")

    except serial.SerialException as e:
        print(f"\n--- Serial Port Error ---")
        print(f"An error occurred: {e}")
        print(f"Please ensure the Nexys 4 board is connected and you are using the correct COM port (currently {SERIAL_PORT}).")

    finally:
        if ser and ser.is_open:
            ser.close()
            print("\nSerial port closed.")

if __name__ == "__main__":
    MODEL_PATH = "cifar10_improved_cnn.pth"
    DATA_BATCH_PATH = "cifar-10-batches-py/data_batch_1" 
    IMAGE_INDEX_TO_TEST = 334
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Load data
    print("Loading CNN model weights and images...")

    try:
        batch = unpickle(DATA_BATCH_PATH)
        image_array = batch[b'data'][IMAGE_INDEX_TO_TEST]
        true_label_index = batch[b'labels'][IMAGE_INDEX_TO_TEST]
        true_label_name = CLASS_NAMES[true_label_index]
    except FileNotFoundError:
        print(f"Error: Data batch not found at '{DATA_BATCH_PATH}'")
        exit()

    
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN(num_classes=len(CLASS_NAMES)).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading model: {e}")
        exit()
    
    # Preprocess image
    image_reshaped = image_array.reshape(3, 32, 32)
    image_tensor = torch.from_numpy(image_reshaped).float() / 255.0
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    input_tensor = normalize(image_tensor).unsqueeze(0).to(device)
    
    # Run prediction on FPGA
    print("Sending image to FPGA...")
    FPGA_SEND(torch.flatten(input_tensor).cpu().numpy())

    # Run control prediction
    print("Running control prediction with pytorch model... ")
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print(f"Model output: {outputs}")

    
