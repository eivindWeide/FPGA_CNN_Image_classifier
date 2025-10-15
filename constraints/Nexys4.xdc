#------------------------------------------------------------------------------------
# Clock Signal
# The board provides a 100MHz clock on pin E3.
#------------------------------------------------------------------------------------
set_property PACKAGE_PIN E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

#------------------------------------------------------------------------------------
# Reset Signal
# Slide switch SW0 is used for the reset signal.
#------------------------------------------------------------------------------------
set_property PACKAGE_PIN U9 [get_ports RST_N]
set_property IOSTANDARD LVCMOS33 [get_ports RST_N]

set_property PACKAGE_PIN U8 [get_ports start_sw]
set_property IOSTANDARD LVCMOS33 [get_ports start_sw]

set_property PACKAGE_PIN R7 [get_ports write_sw]
set_property IOSTANDARD LVCMOS33 [get_ports write_sw]

#------------------------------------------------------------------------------------
# USB-UART Bridge
# Connects the VHDL UART ports to the FTDI chip for serial communication.
# uart_rxd is the FPGA's input (receive).
# uart_txd is the FPGA's output (transmit).
#------------------------------------------------------------------------------------
set_property PACKAGE_PIN D4 [get_ports uart_rxd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rxd]

set_property PACKAGE_PIN C4 [get_ports uart_txd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_txd]

#------------------------------------------------------------------------------------
# Debug LEDs
#------------------------------------------------------------------------------------
set_property PACKAGE_PIN T8 [get_ports {LED[0]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[0]}]
set_property PACKAGE_PIN V9 [get_ports {LED[1]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[1]}]
set_property PACKAGE_PIN R8 [get_ports {LED[2]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[2]}]
set_property PACKAGE_PIN T6 [get_ports {LED[3]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[3]}]
set_property PACKAGE_PIN T5 [get_ports {LED[4]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[4]}]
set_property PACKAGE_PIN T4 [get_ports {LED[5]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[5]}]
set_property PACKAGE_PIN U7 [get_ports {LED[6]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[6]}]
set_property PACKAGE_PIN U6 [get_ports {LED[7]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[7]}]
set_property PACKAGE_PIN V4 [get_ports {LED[8]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[8]}]
set_property PACKAGE_PIN U3 [get_ports {LED[9]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[9]}]
set_property PACKAGE_PIN V1 [get_ports {LED[10]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[10]}]
set_property PACKAGE_PIN R1 [get_ports {LED[11]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[11]}]
set_property PACKAGE_PIN P5 [get_ports {LED[12]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[12]}]
set_property PACKAGE_PIN U1 [get_ports {LED[13]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[13]}]
set_property PACKAGE_PIN R2 [get_ports {LED[14]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[14]}]
set_property PACKAGE_PIN P2 [get_ports {LED[15]}]
    set_property IOSTANDARD LVCMOS33 [get_ports {LED[15]}]
    
set_property PACKAGE_PIN K5 [get_ports frame_err]
    set_property IOSTANDARD LVCMOS33 [get_ports frame_err]
set_property PACKAGE_PIN L16 [get_ports busy]
    set_property IOSTANDARD LVCMOS33 [get_ports busy]   
 