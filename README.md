# FPGA CNN Image Classifier

This is a project for a CNN image classifier using the following layer types on an FPGA: 
- 2d convolution with kernel_size = 3 and padding = 1
- BatchNorm2d
- Maxpooling with kernel_size = 2 and padding = 2
- Fully connected layers
- Rectified linear units

## How to create project in Vivado
1. Clone the repository: `git clone https://github.com/eivindWeide/FPGA_CNN_Image_classifier`
2. Open the Vivado Tcl Shell.
3. `cd` into the directory of the cloned repo.
4. Run the build script: `source ./Nexys4_inference.tcl`
5. Vivado will create the complete project. You can open `Nexys4_inference/my_fpga_project.xpr` in the Vivado GUI.

## Details

The trained model `cifar10_improved_cnn.pth` is pre-loaded on to the FPGAs memory during programming. The model is trained on the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and the layer settings can be found in `extract_weights_from_model.py`. The model has an accuracy of 81% and 114 186 parameters.

To limit memory usage the BatchNorm2d parameters are fused into the convolution weights in `extract_weights_from_model.py` and maxpooling is done concurrently during convolution to reduce the maximum size of the stored feature map by 4x.

The classifier uses the serial port to receive and send data. Using `serial_com.py` it is possible to send an image to the FPGA in the form of 3\*32\*32 \(RGB 32x32 image\) single-precision floating-point numbers and then the FPGA will send the result back. The process is detailed in `serial_com.py`.

Running `serial_com.py` after connecting and programming the FPGA looks like this:

https://github.com/user-attachments/assets/f4b6cb52-3358-45a5-8dc0-ea2cfbf1f8f5

## How to load another model
Changing the model is possible but requires changing a few things in `top_controller.vhd` and in the Vivado GUI. The principal steps are:
1. Add instantiations of layers in the top file `top_controller.vhd`
2. Change the size of bram_din and bram_dout to accommodate the maximum size of the models feature map using Vivado GUI
3. Configure bram_din and bram_dout address multiplexer in `top_controller.vhd`
4. Load the weights from each layer on to the memory using Vivado block memory generator, weights can be extracted using `extract_weights_from_model.py`
5. Add a state for each layer in the *testing* process in `top_controller.vhd`
