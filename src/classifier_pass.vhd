library IEEE;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity classifier_pass is
    generic (
        IN_FEATURES  : natural := 2048;
        OUT_FEATURES : natural := 10;
        DATA_WIDTH   : natural := 32;
        INPUT_MAP_ADDR_SIZE  : natural := 13;
        WEIGHT_MAP_ADDR_SIZE : natural := 15;
        OUTPUT_MAP_ADDR_SIZE : natural := 14
    );
    port (
        -- Control Signals
        clk   : in std_logic;
        rst   : in std_logic;
        start : in std_logic;
        done  : out std_logic;
        
        -- BRAM Interface
        brami_addr : out std_logic_vector(INPUT_MAP_ADDR_SIZE-1 downto 0);
        bramw_addr : out std_logic_vector(WEIGHT_MAP_ADDR_SIZE-1 downto 0);
        bramo_addr : out std_logic_vector(OUTPUT_MAP_ADDR_SIZE-1 downto 0);
        
        brami_dout, bramw_dout : in std_logic_vector(DATA_WIDTH-1 downto 0);
        bramo_wea : out std_logic_vector(0 downto 0);
        bramo_din : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end classifier_pass;

architecture Behavioral of classifier_pass is

    -- Constants
    constant WEIGHT_MAP_SIZE : natural := IN_FEATURES*OUT_FEATURES + OUT_FEATURES;
    
    -- Mac unit registers
    signal mac_ena, mac_clear, mac_ready : std_logic;
    signal mac_data_a, mac_data_b, mac_sum_out : std_logic_vector(DATA_WIDTH-1 downto 0);
    
    -- Read states
    type read_state_t is (R_IDLE, R_INCREMENT, R_FIRST, R_READING, R_WAIT_FOR_MAC);
    signal read_state : read_state_t := R_IDLE;
    
    -- Capture states
    type capture_state_t is (C_IDLE, C_INIT, C_CAPTURING, C_CAPTURE_BIAS);
    signal capture_state : capture_state_t := C_IDLE; 
    
    -- Internal control signals
    signal capture_start, mac_finish : std_logic;
    
    -- Counters
    signal in_feature_cnt  : natural range 0 to IN_FEATURES;
    signal out_feature_cnt : natural range 0 to OUT_FEATURES;
    signal capture_cnt  : natural range 0 to IN_FEATURES;
    signal write_cnt : natural range 0 to OUT_FEATURES;

begin
    
    mac_unit2 : entity work.mac_unit
    port map (
        clk => clk,
        rst => rst,
        ena => mac_ena,
        clear_sum => mac_clear,
        data_a => mac_data_a,
        data_b => mac_data_b,
        ready => mac_ready,
        sum_out => mac_sum_out
    );
    
    
    
    
    -- Three processes, one for setting BRAM address, one for capturing bram_dout to mac unit, and one for writing result from mac unit to bramo
    read : process(clk)
    
    begin
        if rising_edge(clk) then
            if rst = '1' then
                -- Reset
                read_state <= R_IDLE;
                in_feature_cnt  <= 0;
                out_feature_cnt <= 0;
            else
                -- Defaults
                capture_start <= '0';
                brami_addr <= std_logic_vector(TO_UNSIGNED(in_feature_cnt, INPUT_MAP_ADDR_SIZE));
                bramw_addr <= std_logic_vector(TO_UNSIGNED(IN_FEATURES*out_feature_cnt + in_feature_cnt, WEIGHT_MAP_ADDR_SIZE));
                
                case read_state is 
                    when R_IDLE =>
                        if start = '1' then
                            in_feature_cnt  <= 0;
                            out_feature_cnt <= 0;
                            read_state <= R_FIRST;
                        end if;
                    when R_INCREMENT =>
                        read_state <= R_FIRST;
                        
                        out_feature_cnt <= out_feature_cnt + 1;
                        
                        if out_feature_cnt = OUT_FEATURES-1 then
                            -- All weights and features are read
                            out_feature_cnt  <= 0;
                            read_state <= R_IDLE;
                        end if;
                        
                    when R_FIRST => -- Special read state for starting the capture process
                        read_state <= R_READING;
                        capture_start <= '1';
                        in_feature_cnt <= in_feature_cnt + 1;
                    
                    when R_READING =>
                        
                        in_feature_cnt <= in_feature_cnt + 1;
                        
                        if in_feature_cnt = IN_FEATURES-1 then
                            -- All weights and in features for an out feature are read - read bias and wait for mac to finish
                            bramw_addr <= std_logic_vector(TO_UNSIGNED(IN_FEATURES*OUT_FEATURES + out_feature_cnt, WEIGHT_MAP_ADDR_SIZE));
                            read_state <= R_WAIT_FOR_MAC;
                        end if;
                        
                    when R_WAIT_FOR_MAC =>
                        if mac_finish = '1' then
                            read_state <= R_INCREMENT;
                            in_feature_cnt <= 0;
                        end if;
                            
                    when others =>
                        null;
                end case;
            end if;
        end if;        
    end process read;
    
    capture : process(clk)
    
    begin
        if rising_edge(clk) then
            if rst = '1' then
                -- Reset
                capture_state <= C_IDLE;
                capture_cnt  <= 0;
            else
                -- Defaults
                mac_ena <= '0';
                mac_clear <= '0';
                mac_data_a <= brami_dout;
                mac_data_b <= bramw_dout;
                
                case capture_state is
                    when C_IDLE =>
                        if capture_start = '1' then
                            mac_clear <= '1';
                            capture_cnt  <= 0;
                            capture_state <= C_INIT;
                        end if;
                    when C_INIT =>
                        -- wait for one cycle for bram_dout to be ready
                        capture_state <= C_CAPTURING;
                        capture_cnt <= capture_cnt + 1;
                    
                    when C_CAPTURING =>
                        capture_cnt <= capture_cnt + 1;
                        
                        mac_ena <= '1';
                        
                        if capture_cnt = IN_FEATURES-1 then
                            -- All weights and in features for an out feature captured
                            capture_state <= C_CAPTURE_BIAS;
                            capture_cnt <= 0;
                        end if;
                        
                    when C_CAPTURE_BIAS =>
                        -- Capture trailing bias for out feature
                        mac_ena <= '1';
                        mac_data_a <= x"3f800000"; -- 1
                        capture_state <= C_IDLE;
                        
                    when others =>
                        null;
                end case;
            end if;
        end if;
    end process capture;
    
    write : process(clk)
    
    begin
        if rising_edge(clk) then
            if rst = '1' then
                -- Reset
                write_cnt <= 0;
            else
                -- Defaults
                done <= '0';
                mac_finish <= '0';
                bramo_wea <= "0";
                bramo_addr <= std_logic_vector(TO_UNSIGNED(write_cnt, OUTPUT_MAP_ADDR_SIZE));
                
                if write_cnt = OUT_FEATURES then
                    write_cnt <= 0;
                    done <= '1';
                elsif mac_ready = '1' then
                    bramo_wea <= "1";
                    write_cnt <= write_cnt + 1;
                    mac_finish <= '1';
                end if;
            end if;
        end if;
    end process write;
    
    bramo_din <= mac_sum_out;

end Behavioral;
