library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity top_controller_tb is
end top_controller_tb;

architecture behavioral of top_controller_tb is

    -- Component Declaration for the Device Under Test (DUT)
    component top_controller is
        port (
            clk      : in  std_logic;
            rst_n    : in  std_logic;
            uart_rxd : in  std_logic;
            uart_txd : out std_logic;
            LED : out std_logic_vector(15 downto 0);
            
            start_sw : in std_logic;
            write_sw : in std_logic
        );
    end component;

    -- Constants
    constant CLK_FREQ     : integer := 100_000_000;
    constant CLK_PERIOD   : time    := 1 sec / CLK_FREQ;
    constant BAUD_RATE    : integer := 115_200;
    constant BIT_PERIOD   : time    := 1 sec / BAUD_RATE;
    constant NUM_WORDS    : integer := 32;
    constant data_value   : std_logic_vector(7 downto 0) := x"41";

    -- Testbench Signals
    signal clk      : std_logic := '0';
    signal rst_n      : std_logic;
    signal uart_rxd : std_logic := '0'; -- UART is idle high
    signal uart_txd : std_logic;
    signal LED      : std_logic_vector(15 downto 0);
    signal start_sw : std_logic;
    signal write_sw : std_logic;
    
    -- Signal to hold the current value from test file
    signal file_value : signed(31 downto 0) := (others => '0');
    -- Control signal for the loop
    signal done : boolean := false;
    
    -- Constants for file reading
    constant FILE_NAME : string := "test_image.txt"; 

begin

    -- Instantiate the DUT
    dut : top_controller
        port map (
            clk      => clk,
            rst_n      => rst_n,
            uart_rxd => uart_rxd,
            uart_txd => uart_txd,
            LED => LED,
            start_sw => start_sw,
            write_sw => write_sw
        );

    -- Clock generation process
    clk_process : process
    begin
        clk <= '0';
        wait for CLK_PERIOD / 2;
        clk <= '1';
        wait for CLK_PERIOD / 2;
    end process;

    -- Stimulus process
    stim_proc : process
        
        -- File pointer
        file data_file : TEXT open READ_MODE is FILE_NAME;
        variable file_line : LINE;
        variable hex_value : std_logic_vector(31 downto 0);
        variable char_buffer : character;
    
        -- Procedure to send one byte over UART
        procedure send_byte(data_in : in std_logic_vector(7 downto 0)) is
        begin
            -- Start Bit
            uart_rxd <= '0';
            wait for BIT_PERIOD;
            -- Data Bits (LSB first)
            for i in 0 to 7 loop
                uart_rxd <= data_in(i);
                wait for BIT_PERIOD;
            end loop;
            -- Stop Bit
            uart_rxd <= '1';
            wait for BIT_PERIOD;
        end procedure send_byte;
        
        -- Procedure to send a 32-bit integer as 4 bytes
        procedure send_word(data_in : in std_logic_vector(31 downto 0)) is
        begin
            send_byte(std_logic_vector(data_in(31 downto 24)));
            send_byte(std_logic_vector(data_in(23 downto 16)));
            send_byte(std_logic_vector(data_in(15 downto  8)));
            send_byte(std_logic_vector(data_in( 7 downto  0)));
        end procedure send_word;

        -- Procedure to receive one byte from UART
        procedure receive_byte(data_out : out std_logic_vector(7 downto 0)) is
            variable temp_byte : std_logic_vector(7 downto 0);
            variable L         : line; -- ADD THIS LINE
        begin
            -- Wait for start bit
            wait until uart_txd = '0';
            -- Wait half a bit period to sample in the middle of the bit
            wait for BIT_PERIOD / 2;
            -- Confirm start bit
            assert uart_txd = '0' report "Did not see start bit" severity failure;
            wait for BIT_PERIOD;
            -- Read data bits
            for i in 0 to 7 loop
                temp_byte(i) := uart_txd;
                wait for BIT_PERIOD;
            end loop;
        
            -- ADD THE FOLLOWING 2 LINES to report the received byte --
            hwrite(L, temp_byte);
            report "TB RX Byte: " & L.all;
        
            data_out := temp_byte;
            -- Check for stop bit
            assert uart_txd = '1' report "Did not see stop bit" severity failure;
        end procedure receive_byte;
        
        -- Procedure to receive a 32-bit integer as 4 bytes
        procedure receive_word(data_out : out std_logic_vector(31 downto 0)) is
             variable byte0, byte1, byte2, byte3 : std_logic_vector(7 downto 0);
        begin
             receive_byte(byte3);
             receive_byte(byte2);
             receive_byte(byte1);
             receive_byte(byte0);
             data_out := (byte3 & byte2 & byte1 & byte0);
        end procedure receive_word;

        variable received_word : signed(31 downto 0);
        variable expected_word : signed(31 downto 0);
        variable error_count   : integer := 0;

    begin
        -- 1. Apply Reset
        start_sw <= '0';
        write_sw <= '0';
        uart_rxd <= '1';
		RST_N <= '0';
		wait for 100 ns;
    	RST_N <= '1';
        
        wait for BIT_PERIOD;
        
        uart_rxd <= '0'; -- start bit
		wait for BIT_PERIOD;
        
        for i in 0 to (data_value'LENGTH-1) loop
		    uart_rxd <= data_value(i); -- data bits
		    wait for BIT_PERIOD;
		end loop;

		uart_rxd <= '1'; -- stop bit
		wait for BIT_PERIOD;
        
        -- 2. Write data to BRAM1
        report "Writing to BRAM1...";
        
        for I in 0 to 3072-1 loop
            if not endfile(data_file) then
                readline(data_file, file_line);
               
                -- Read the 8-character Hex value from the line
                hread(file_line, hex_value);
                
                send_word(hex_value);
            end if;
        end loop;
        report "Finished writing to BRAM1.";
        wait; -- End of simulation
    end process stim_proc;

end behavioral;
