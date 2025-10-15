# FPGA CNN Image Classifier

This is a project for a CNN image classifier implemented on an FPGA. The model implemented is trained on the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

Using the serial_com.py script it is possible to send an image to the FPGA in the form of 3*32*32 floating-point numbers and then the FPGA will send the result back over the serial port. The process is specified in the file. 

Running the script looks like this:

https://github.com/user-attachments/assets/f4b6cb52-3358-45a5-8dc0-ea2cfbf1f8f5

## How to create project in Vivado
1. Clone the repository: `git clone https://github.com/eivindWeide/FPGA_CNN_Image_classifier`
2. Open the Vivado Tcl Shell.
3. `cd` into the directory of the cloned repo.
4. Run the build script: `source ./Nexys4_inference.tcl`
5. Vivado will create the complete project. You can open `Nexys4_inference/my_fpga_project.xpr` in the Vivado GUI.
