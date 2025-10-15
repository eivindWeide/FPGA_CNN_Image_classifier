library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top_controller is
    Generic (
        CLK_FREQ   : integer := 100e6;   -- set system clock frequency in Hz
        BAUD_RATE  : integer := 115200; -- baud rate value
        PARITY_BIT : string  := "none"  -- legal values: "none", "even", "odd", "mark", "space"
    );
    port (
        clk      : in  std_logic;
        RST_N      : in  std_logic;
        start_sw : in std_logic;
        write_sw : in std_logic;
        
        -- UART Interface
        uart_rxd  : in  std_logic;
        uart_txd  : out std_logic;
        busy      : out std_logic;
        frame_err : out std_logic;
        
        LED : out std_logic_vector(15 downto 0)
    );
end top_controller;

architecture behavioral of top_controller is
    -- State machine definition
    type state_t is (
        S_IDLE, 
        S_RX_BRAM1,
        S_TX_READ_BRAM3,
        S_TX_BRAM3_WAIT1,
        S_TX_BRAM3_WAIT2,
        S_TX_BRAM3
    );
    signal current_state : state_t := S_IDLE;

    type testing_state_t is (TESTING_IDLE, TESTING_COMPUTE1, TESTING_COMPUTE2, TESTING_COMPUTE3, TESTING_COMPUTE4, TESTING_DONE, TESTING_ERROR);
    signal testing_state : testing_state_t := TESTING_IDLE;
    
    type uart_state_t is (UART_IDLE, UART_START, UART_WRITE, UART_DONE);
    signal uart_state : uart_state_t := UART_IDLE;

    signal rst : std_logic;

    -- UART component signals
    signal data_out, data_in    : std_logic_vector(7 downto 0);
    signal tx_valid, rx_valid   : std_logic;
    signal r_busy                 : std_logic;
    
    signal brami_addr : std_logic_vector(11 downto 0);
    signal brami_ena : std_logic;
    signal brami_wea : std_logic_vector(0 downto 0);
    signal brami_dout : std_logic_vector(31 downto 0);
    signal brami_din : std_logic_vector(31 downto 0);
    
    signal bramo_addr : std_logic_vector(12 downto 0);
    signal bramo_ena : std_logic;
    signal bramo_wea : std_logic_vector(0 downto 0);
    signal bramo_dout : std_logic_vector(31 downto 0);
    signal bramo_din : std_logic_vector(31 downto 0);
    
    signal bramw_addr : std_logic_vector(9 downto 0);
    signal bramw_ena : std_logic;
    signal bramw_wea : std_logic_vector(0 downto 0);
    signal bramw_dout : std_logic_vector(31 downto 0);
    signal bramw_din : std_logic_vector(31 downto 0);
    
    signal bramw2_addr : std_logic_vector(14 downto 0);
    signal bramw2_ena : std_logic;
    signal bramw2_wea : std_logic_vector(0 downto 0);
    signal bramw2_dout : std_logic_vector(31 downto 0);
    signal bramw2_din : std_logic_vector(31 downto 0);
    
    signal bramw3_addr : std_logic_vector(16 downto 0);
    signal bramw3_ena : std_logic;
    signal bramw3_wea : std_logic_vector(0 downto 0);
    signal bramw3_dout : std_logic_vector(31 downto 0);
    signal bramw3_din : std_logic_vector(31 downto 0);
    
    signal bramw4_addr : std_logic_vector(14 downto 0);
    signal bramw4_ena : std_logic;
    signal bramw4_wea : std_logic_vector(0 downto 0);
    signal bramw4_dout : std_logic_vector(31 downto 0);
    signal bramw4_din : std_logic_vector(31 downto 0);
    
    signal block1_brami_addr, block2_bramo_addr, block3_brami_addr, classifier_bramo_addr, rx_brami_addr : std_logic_vector(11 downto 0);
    signal block1_bramo_addr, block2_brami_addr, block3_bramo_addr, classifier_brami_addr : std_logic_vector(12 downto 0);
    signal block1_bramo_din, block2_bramo_din, block3_bramo_din, classifier_bramo_din, rx_brami_din : std_logic_vector(31 downto 0);
    signal block1_bramo_wea, block2_bramo_wea, block3_bramo_wea, classifier_bramo_wea, rx_brami_wea : std_logic_vector(0 downto 0);

    -- Internal registers and counters for word assembly
    signal byte_counter  : integer range 0 to 4 := 0;
    signal word_counter  : integer range 0 to 3071 := 0;
    signal byte1, byte2, byte3 : std_logic_vector(7 downto 0);
    
    -- Signals for feature_block_pass
    signal start_compute, start_compute1, start_compute2, start_compute3, start_compute4, no_compute : std_logic := '0';
    signal compute_done, compute_done2, compute_done3, compute_done4  : std_logic;
    signal compute_finish : std_logic;
    
    -- Write array
    type write_array_t is array (0 to 39) of std_logic_vector(7 downto 0);
    signal write_array : write_array_t := (
        x"41", x"20", x"00", x"00",
        x"41", x"30", x"00", x"00",
        x"41", x"40", x"00", x"00",
        x"41", x"50", x"00", x"00",
        x"41", x"60", x"00", x"00",
        x"41", x"70", x"00", x"00",
        x"41", x"80", x"00", x"00",
        x"41", x"88", x"00", x"00",
        x"41", x"90", x"00", x"00",
        x"41", x"98", x"00", x"00"
    );
    signal write_cnt : natural range 0 to 39 := 0;
    
    COMPONENT bram_din
      PORT (
        clka : IN STD_LOGIC;
        ena : IN STD_LOGIC;
        wea : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
        addra : IN STD_LOGIC_VECTOR(11 DOWNTO 0);
        dina : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
        douta : OUT STD_LOGIC_VECTOR(31 DOWNTO 0) 
      );
    END COMPONENT;
    
    COMPONENT bramw1
      PORT (
        clka : IN STD_LOGIC;
        ena : IN STD_LOGIC;
        wea : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
        addra : IN STD_LOGIC_VECTOR(9 DOWNTO 0);
        dina : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
        douta : OUT STD_LOGIC_VECTOR(31 DOWNTO 0) 
      );
    END COMPONENT;
    
    COMPONENT bram_dout
      PORT (
        clka : IN STD_LOGIC;
        ena : IN STD_LOGIC;
        wea : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
        addra : IN STD_LOGIC_VECTOR(12 DOWNTO 0);
        dina : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
        douta : OUT STD_LOGIC_VECTOR(31 DOWNTO 0) 
      );
    END COMPONENT;
    
    COMPONENT bramw2
      PORT (
        clka : IN STD_LOGIC;
        ena : IN STD_LOGIC;
        wea : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
        addra : IN STD_LOGIC_VECTOR(14 DOWNTO 0);
        dina : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
        douta : OUT STD_LOGIC_VECTOR(31 DOWNTO 0) 
      );
    END COMPONENT;
    
    COMPONENT bramw3
      PORT (
        clka : IN STD_LOGIC;
        ena : IN STD_LOGIC;
        wea : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
        addra : IN STD_LOGIC_VECTOR(16 DOWNTO 0);
        dina : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
        douta : OUT STD_LOGIC_VECTOR(31 DOWNTO 0) 
      );
    END COMPONENT;
    
    COMPONENT bramw4
      PORT (
        clka : IN STD_LOGIC;
        ena : IN STD_LOGIC;
        wea : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
        addra : IN STD_LOGIC_VECTOR(14 DOWNTO 0);
        dina : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
        douta : OUT STD_LOGIC_VECTOR(31 DOWNTO 0) 
      );
    END COMPONENT;
    
begin

    rst <= not RST_N;
    busy <= r_busy;
    
 
    
 
    -- Instantiate UART
    uart_i : entity work.uart
    generic map (
        CLK_FREQ => CLK_FREQ,
        BAUD_RATE => BAUD_RATE,
        PARITY_BIT => PARITY_BIT
    )
    port map (
        CLK         => CLK,
        RST         => rst,
        -- UART INTERFACE
        UART_TXD    => UART_TXD,
        UART_RXD    => UART_RXD,
        -- USER DATA OUTPUT INTERFACE
        DATA_OUT    => data_in,
        DATA_VLD    => rx_valid,
        FRAME_ERROR => FRAME_ERR,
        -- USER DATA INPUT INTERFACE
        DATA_IN     => data_out,
        DATA_SEND   => tx_valid,
        BUSY        => r_busy
    );

    -- Instantiate BRAMs  
    brami : bram_din
      PORT MAP (
        clka => clk,
        ena => brami_ena,
        wea => brami_wea,
        addra => brami_addr,
        dina => brami_din,
        douta => brami_dout
      );
      
      bramw : bramw1
      PORT MAP (
        clka => clk,
        ena => bramw_ena,
        wea => bramw_wea,
        addra => bramw_addr,
        dina => bramw_din,
        douta => bramw_dout
      );
      
      bramo : bram_dout
      PORT MAP (
        clka => clk,
        ena => bramo_ena,
        wea => bramo_wea,
        addra => bramo_addr,
        dina => bramo_din,
        douta => bramo_dout
      );
      
      bram2 : bramw2
      PORT MAP (
        clka => clk,
        ena => bramw2_ena,
        wea => bramw2_wea,
        addra => bramw2_addr,
        dina => bramw2_din,
        douta => bramw2_dout
      );
      
      bram3 : bramw3
      PORT MAP (
        clka => clk,
        ena => bramw3_ena,
        wea => bramw3_wea,
        addra => bramw3_addr,
        dina => bramw3_din,
        douta => bramw3_dout
      );
      
      bram4 : bramw4
      PORT MAP (
        clka => clk,
        ena => bramw4_ena,
        wea => bramw4_wea,
        addra => bramw4_addr,
        dina => bramw4_din,
        douta => bramw4_dout
      );
    
    -- Instantiate feature block unit
    feature_block_pass : entity work.feature_block_pass
    generic map (
        IN_CHANNELS          => 3,
        OUT_CHANNELS         => 32,
        IN_SIZE              => 32,
        OUT_SIZE             => 16,
        DATA_WIDTH           => 32,
        KERNEL_SIZE          => 3,
        INPUT_MAP_ADDR_SIZE  => 12,
        WEIGHT_MAP_ADDR_SIZE => 10,
        OUTPUT_MAP_ADDR_SIZE => 13
    )
    port map (
        clk        => clk,
        rst        => rst,
        start      => start_compute1,
        done       => compute_done,
        brami_dout => brami_dout,
        bramw_dout => bramw_dout,
        brami_addr => block1_brami_addr,
        bramw_addr => bramw_addr,
        bramo_wea   => block1_bramo_wea,
        bramo_addr => block1_bramo_addr,
        bramo_din  => block1_bramo_din
    );
    
    feature_block_pass2 : entity work.feature_block_pass
    generic map (
        IN_CHANNELS          => 32,
        OUT_CHANNELS         => 64,
        IN_SIZE              => 16,
        OUT_SIZE             => 8,
        DATA_WIDTH           => 32,
        KERNEL_SIZE          => 3,
        INPUT_MAP_ADDR_SIZE  => 13,
        WEIGHT_MAP_ADDR_SIZE => 15,
        OUTPUT_MAP_ADDR_SIZE => 12
    )
    port map (
        clk        => clk,
        rst        => rst,
        start      => start_compute2,
        done       => compute_done2,
        brami_dout => bramo_dout,
        bramw_dout => bramw2_dout,
        brami_addr => block2_brami_addr,
        bramw_addr => bramw2_addr,
        bramo_wea   => block2_bramo_wea,
        bramo_addr => block2_bramo_addr,
        bramo_din  => block2_bramo_din
    );
    
    feature_block_pass3 : entity work.feature_block_pass
    generic map (
        IN_CHANNELS          => 64,
        OUT_CHANNELS         => 128,
        IN_SIZE              => 8,
        OUT_SIZE             => 4,
        DATA_WIDTH           => 32,
        KERNEL_SIZE          => 3,
        INPUT_MAP_ADDR_SIZE  => 12,
        WEIGHT_MAP_ADDR_SIZE => 17,
        OUTPUT_MAP_ADDR_SIZE => 13
    )
    port map (
        clk        => clk,
        rst        => rst,
        start      => start_compute3,
        done       => compute_done3,
        brami_dout => brami_dout,
        bramw_dout => bramw3_dout,
        brami_addr => block3_brami_addr,
        bramw_addr => bramw3_addr,
        bramo_wea   => block3_bramo_wea,
        bramo_addr => block3_bramo_addr,
        bramo_din  => block3_bramo_din
    );
    
    classifier_pass : entity work.classifier_pass
    generic map (
        IN_FEATURES => 2048,
        OUT_FEATURES => 10,
        DATA_WIDTH => 32,
        INPUT_MAP_ADDR_SIZE => 13,
        WEIGHT_MAP_ADDR_SIZE => 15,
        OUTPUT_MAP_ADDR_SIZE => 12
    )
    port map (
        clk        => clk,
        rst        => rst,
        start      => start_compute4,
        done       => compute_done4,
        brami_dout => bramo_dout,
        bramw_dout => bramw4_dout,
        brami_addr => classifier_brami_addr,
        bramw_addr => bramw4_addr,
        bramo_wea  => classifier_bramo_wea,
        bramo_addr => classifier_bramo_addr,
        bramo_din  => classifier_bramo_din
    );
    
    
    -- BRAM multiplexers                 
    with testing_state select
        brami_addr <=   block1_brami_addr       when TESTING_COMPUTE1,
                        block2_bramo_addr       when TESTING_COMPUTE2,
                        block3_brami_addr       when TESTING_COMPUTE3,
                        classifier_bramo_addr   when TESTING_COMPUTE4,
                        rx_brami_addr           when others;
                        
    with testing_state select
        brami_wea  <=   block2_bramo_wea     when TESTING_COMPUTE2,
                        classifier_bramo_wea when TESTING_COMPUTE4,
                        rx_brami_wea     when others;
                        
    with testing_state select
        brami_din  <=   block2_bramo_din     when TESTING_COMPUTE2,
                        classifier_bramo_din when TESTING_COMPUTE4,
                        rx_brami_din     when others;                    
                        
    with testing_state select
        bramo_addr <=   block1_bramo_addr       when TESTING_COMPUTE1,
                        block2_brami_addr       when TESTING_COMPUTE2,
                        block3_bramo_addr       when TESTING_COMPUTE3,
                        classifier_brami_addr   when TESTING_COMPUTE4,
                        block1_bramo_addr       when others;
                        
    with testing_state select 
        bramo_wea  <=   block1_bramo_wea when TESTING_COMPUTE1,
                        block3_bramo_wea when TESTING_COMPUTE3,
                        "0"              when others;
                        
    with testing_state select 
        bramo_din <=    block1_bramo_din when TESTING_COMPUTE1,
                        block3_bramo_din when TESTING_COMPUTE3,
                        block1_bramo_din when others;
                      
    brami_ena <= '1';
    bramo_ena <= '1';
    bramw_ena <= '1';
    bramw2_ena <= '1';
    bramw3_ena <= '1';
    bramw4_ena <= '1';
    bramw_wea <= "0";
    bramw2_wea <= "0";
    bramw3_wea <= "0";
    bramw4_wea <= "0";
    bramw_din <= (others => '0');
    bramw2_din <= (others => '0');
    bramw3_din <= (others => '0');
    bramw4_din <= (others => '0');
    
    
    -- UART Control FSM
    process(clk)
    begin
        if(rising_edge(clk)) then
            if RST_N = '0' then
                uart_state <= UART_IDLE;
                tx_valid <= '0';
                write_cnt <= 0;
            else
                tx_valid <= '0'; -- default: no transmit
                data_out <= x"00";
                case uart_state is
                    when UART_IDLE =>
                        if rx_valid = '1' then
                            if data_in = x"41" then
                                uart_state <= UART_START;
                           elsif data_in = x"42" then
                                write_cnt <= 0;
                                uart_state <= UART_WRITE;
                            else 
                                no_compute <= '1';
                                uart_state <= UART_IDLE;
                            end if;
                        end if;
                        
                    when UART_START =>
                        if compute_finish = '1' then
                            uart_state <= UART_IDLE;
                        end if;
                        
                    when UART_WRITE =>
                        if r_busy = '0' and tx_valid = '0' then
                            tx_valid <= '1';
                            data_out <= write_array(write_cnt);
                            if write_cnt >= 39 then
                                uart_state <= UART_DONE;
                            else
                                write_cnt <= write_cnt + 1;
                            end if;
                        end if;
                        
                    when UART_DONE =>
                        null;
                        
                 end case;
            end if;
        end if;
    end process;

    -- Main Control FSM
    process(clk)
    begin
        if rising_edge(clk) then
            if RST_N = '0' then
                current_state <= S_IDLE;
                byte_counter <= 0; 
                word_counter <= 0;
                write_array <= (
                    x"41", x"20", x"00", x"00",
                    x"41", x"30", x"00", x"00",
                    x"41", x"40", x"00", x"00",
                    x"41", x"50", x"00", x"00",
                    x"41", x"60", x"00", x"00",
                    x"41", x"70", x"00", x"00",
                    x"41", x"80", x"00", x"00",
                    x"41", x"88", x"00", x"00",
                    x"41", x"90", x"00", x"00",
                    x"41", x"98", x"00", x"00"
                );
                
            else
                -- Defaults
                start_compute <= '0';
                rx_brami_wea <= "0";
                rx_brami_addr <= std_logic_vector(to_unsigned(word_counter, 12));
                rx_brami_din <= x"42280000";
                case current_state is
                
                    -- Wait for a command byte
                    when S_IDLE =>
                        if rx_valid = '1' then
                            if data_in = x"41" then
                                current_state <= S_RX_BRAM1;
                                byte_counter <= 0;
                                word_counter <= 0;
                           end if;
                        elsif compute_finish = '1' then
                            current_state <= S_TX_READ_BRAM3;
                            byte_counter <= 0;
                            word_counter <= 0;
                        end if;

                    -- Receive 3072 bytes for BRAM1
                    when S_RX_BRAM1 =>
                        if rx_valid = '1' then
                            case byte_counter is
                                when 0 => byte3 <= data_in; -- MSB (Big-Endian)
                                when 1 => byte2 <= data_in;
                                when 2 => byte1 <= data_in;
                                when 3 => 
                                    -- On the 4th byte, trigger the write
                                    rx_brami_wea  <= "1";
                                    rx_brami_din  <= byte3 & byte2 & byte1 & data_in;
                                when others => null;
                            end case;
                            
                            -- Always update the counters
                            if byte_counter = 3 then
                                byte_counter <= 0;
                                if word_counter = 3071 then
                                    start_compute <= '1';
                                    current_state <= S_IDLE;
                                else
                                    word_counter <= word_counter + 1;
                                end if;
                            else
                                byte_counter <= byte_counter + 1;
                            end if;
                        end if;
                        
                        
                    when S_TX_READ_BRAM3 =>
                        current_state <= S_TX_BRAM3_WAIT1;
                        
                    when S_TX_BRAM3_WAIT1 =>
                        current_state <= S_TX_BRAM3_WAIT2;
                    
                    when S_TX_BRAM3_WAIT2 =>
                        current_state <= S_TX_BRAM3;

                    -- Save first 10 BRAMI contents to write array
                    when S_TX_BRAM3 =>
                    
                        write_array(word_counter*4 + 0) <= brami_dout(31 downto 24);
                        write_array(word_counter*4 + 1) <= brami_dout(23 downto 16);
                        write_array(word_counter*4 + 2) <= brami_dout(15 downto 8);
                        write_array(word_counter*4 + 3) <= brami_dout(7 downto 0);
                        
                        if word_counter >= 9 then
                            current_state <= S_IDLE;
                        else
                            word_counter <= word_counter + 1;
                            current_state <= S_TX_READ_BRAM3; 
                        end if;
                end case;
            end if;
        end if;
    end process;
    
    testing : process(clk)
    
    begin
    if rising_edge(clk) then
            if RST_N = '0' then
                testing_state <= TESTING_IDLE;
            else 
                LED <= "0000000000000001";
                start_compute1 <= '0';
                start_compute2 <= '0';
                start_compute3 <= '0';
                start_compute4 <= '0';
                compute_finish <= '0';
                
                case testing_state is 
                    when TESTING_IDLE =>
                        LED <= "0000010101000001";
                        if start_compute = '1' then
                            testing_state <= TESTING_COMPUTE1;
                            start_compute1 <= '1';
                        elsif no_compute = '1' then
                            testing_state <= TESTING_ERROR;
                        end if;
                        
                    when TESTING_COMPUTE1 =>
                        LED <= "0010100000001100";
                        if compute_done = '1' then
                            start_compute2 <= '1';
                            testing_state <= TESTING_COMPUTE2;
                        end if;
                    when TESTING_COMPUTE2 =>
                        LED <= "0001000000001111";
                        if compute_done2 = '1' then
                            start_compute3 <= '1';
                            testing_state <= TESTING_COMPUTE3;
                        end if;
                        
                    when TESTING_COMPUTE3 =>
                        LED <= "0011111000000000";
                        if compute_done3 = '1' then
                            start_compute4 <= '1';
                            testing_state <= TESTING_COMPUTE4;
                        end if;
                        
                    when TESTING_COMPUTE4 =>
                        LED <= "0000010000000000";
                        if compute_done4 = '1' then
                            compute_finish <= '1';
                            testing_state <= TESTING_DONE;
                        end if;
                        
                    when TESTING_DONE =>
                        LED <= "1111111111111111";
                        
                    when TESTING_ERROR =>
                        LED <= "1010101010101010";
                    
                end case;
            end if;
        end if;
    end process testing;
end behavioral;

