library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity feature_block_pass is
    generic (
        IN_CHANNELS          : natural := 3;
        OUT_CHANNELS         : natural := 32;
        IN_SIZE              : natural := 32;
        OUT_SIZE             : natural := 16;
        DATA_WIDTH           : natural := 32;
        KERNEL_SIZE          : natural := 3;
        INPUT_MAP_ADDR_SIZE  : natural := 12;
        WEIGHT_MAP_ADDR_SIZE : natural := 10;
        OUTPUT_MAP_ADDR_SIZE : natural := 13
    );
    port (
        -- Control Signals
        clk   : in  std_logic;
        rst   : in  std_logic;
        start : in  std_logic;
        done  : out std_logic;

        -- BRAM Interface
        brami_addr : out std_logic_vector(INPUT_MAP_ADDR_SIZE-1 downto 0);
        bramw_addr : out std_logic_vector(WEIGHT_MAP_ADDR_SIZE-1 downto 0);
        bramo_addr : out std_logic_vector(OUTPUT_MAP_ADDR_SIZE-1 downto 0);
        
        brami_dout, bramw_dout : in std_logic_vector(DATA_WIDTH-1 downto 0);
        bramo_wea : out std_logic_vector(0 downto 0);
        bramo_din : out std_logic_vector(DATA_WIDTH-1 downto 0)
    );
end feature_block_pass;

architecture behavioral of feature_block_pass is

    -- constants
    constant IN_SIZE_SQ : natural := IN_SIZE**2;
    constant OUT_SIZE_SQ : natural := OUT_SIZE**2;
    constant KERNEL_SIZE_SQ : natural := KERNEL_SIZE**2;
    constant INPUT_MAP_SIZE : natural := IN_CHANNELS*IN_SIZE_SQ;
    constant WEIGHT_MAP_SIZE : natural := OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE_SQ + OUT_CHANNELS; -- Includes biases at the end
    constant OUTPUT_MAP_SIZE : natural := OUT_CHANNELS*OUT_SIZE_SQ;
    constant WEIGHT_STRIDE : natural := IN_CHANNELS*KERNEL_SIZE_SQ; -- stride needed to jump between filters in the weights map.
    
    -- Lookup table for index offsets of all in-features needed in one convolution step (read backwards)
    type in_feature_offset_table_t is array (0 to KERNEL_SIZE_SQ-1) of natural;
    constant IN_FEATURE_OFFSET_TABLE : in_feature_offset_table_t := (
        -IN_SIZE - 1, -IN_SIZE, -IN_SIZE + 1,
                 - 1,        0,          + 1,
         IN_SIZE - 1,  IN_SIZE,  IN_SIZE + 1
    );
    
    -- Lookup table for index offsets for in-feature pooling groups
    type pooling_group_offset_table_t is array (0 to 3) of natural;
    constant POOLING_GROUP_OFFSET_TABLE : pooling_group_offset_table_t := (
        0*IN_SIZE + 0, 0*IN_SIZE + 1,
        1*IN_SIZE + 0, 1*IN_SIZE + 1
    );
    
    -- control states
    type c_state_t is (IDLE, INCREMENT, SET_PADDING, READ_FIRST, READING, WAIT_FOR_MAC);
    signal current_state : c_state_t := IDLE;
    
    -- read states
    type r_state_t is (IDLE, INIT, CAPTURE, CAPTURE_BIAS);
    signal r_current_state : r_state_t := IDLE;
    
    -- state control signals
    signal read_start, mac_finish : std_logic;
    
    -- Array for setting padding
    type padding_mask_t is array (0 to KERNEL_SIZE_SQ-1) of std_logic;
    signal padding_mask : padding_mask_t;
    
    -- array for storing results from mac before pooling
    type mac_results_t is array (0 to 3) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mac_results : mac_results_t;
    signal pool_result : std_logic_vector(31 downto 0);
    signal pool_rdy, pool_en : std_logic;
    
    -- signals for traversing the input feature map
    signal out_chn_counter : integer range 0 to OUT_CHANNELS := 0;
    signal pixel_counter   : integer range 0 to IN_SIZE_SQ := 0;
    
    -- signal for grouping 4 input features to the same pooling layer
    signal pooling_counter : natural range 0 to 3;
    
    -- mac unit registers
    signal mac_ena     : std_logic;
    signal mac_clear   : std_logic;
    signal mac_data_a  : std_logic_vector(31 downto 0);
    signal mac_data_b  : std_logic_vector(31 downto 0);
    signal mac_data_c  : std_logic_vector(31 downto 0);
    signal mac_ready   : std_logic;
    signal mac_sum_out : std_logic_vector(31 downto 0);
    
    -- Signals converted from variables
    signal weight_map_index_start  : natural range 0 to WEIGHT_MAP_SIZE-1;
    signal weight_map_index_offset : natural range 0 to KERNEL_SIZE_SQ*IN_CHANNELS; -- Range corrected
    signal input_map_index         : natural range 0 to INPUT_MAP_SIZE-1;
    signal input_map_index_offset  : natural range 0 to KERNEL_SIZE_SQ-1;
    signal w_cnt                   : natural range 0 to KERNEL_SIZE_SQ*IN_CHANNELS := 0;
    signal out_cnt                 : natural range 0 to OUT_CHANNELS*OUT_SIZE_SQ := 0;
    signal pool_cnt                : natural range 0 to 4 := 0; -- Range corrected
    
begin  
    mac_unit : entity work.mac_unit
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

    
    comparator : entity work.fp32_comparator
    port map (
        CLK => clk,
        RST => rst,
        EN => pool_en,
        A => mac_results(0),
        B => mac_results(1),
        C => mac_results(2),
        D => mac_results(3),
        ready => pool_rdy,
        Largest_Output => pool_result
    );

    control : process(clk) -- Process for iteration and reading ram.
    
    variable first_index : natural range 0 to INPUT_MAP_SIZE-1;
    
    begin
        if rising_edge(clk) then
            if rst = '1' then
                current_state   <= IDLE;
                out_chn_counter <= 0;
                pixel_counter   <= 0;
                pooling_counter <= 0;
                padding_mask <= ('0','0','0','0','1','1','0','1','1');
                weight_map_index_start  <= 0;
                weight_map_index_offset <= 0;
                input_map_index  <= 0;
                input_map_index_offset  <= 0;
            else
                -- Defaults
                read_start <= '0';
                brami_addr <= std_logic_vector(TO_UNSIGNED(input_map_index + (IN_FEATURE_OFFSET_TABLE(input_map_index_offset)), INPUT_MAP_ADDR_SIZE));
                if weight_map_index_offset = KERNEL_SIZE_SQ*IN_CHANNELS then
                    -- Read bias after all weights are read
                    bramw_addr <= std_logic_vector(TO_UNSIGNED(OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE_SQ + out_chn_counter, WEIGHT_MAP_ADDR_SIZE));
                else 
                    -- Load weights
                    bramw_addr <= std_logic_vector(TO_UNSIGNED(weight_map_index_start + weight_map_index_offset, WEIGHT_MAP_ADDR_SIZE));
                end if;
                
                case current_state is
                    when IDLE =>
                        if start = '1' then
                            current_state   <= SET_PADDING;
                            out_chn_counter <= 0;
                            pixel_counter   <= 0;
                            pooling_counter <= 0;
                            padding_mask    <= ('0','0','0','0','1','1','0','1','1'); -- Starting at top left
                            weight_map_index_start  <= 0;
                            weight_map_index_offset <= 0;
                            input_map_index         <= 0;
                            input_map_index_offset  <= 0;
                        end if;
                        
                    when INCREMENT =>
                        -- for each channel in OUT_CHANNELS:
                            -- for each 'pixel' in OUT_SIZE_SQ:
                
                        pixel_counter <= pixel_counter + 1;
                        current_state <= SET_PADDING;
                        
                        if pixel_counter = OUT_SIZE_SQ-1 then
                            out_chn_counter <= out_chn_counter + 1;
                            pixel_counter <= 0;
                        end if;
                            
                        if pixel_counter = OUT_SIZE_SQ-1
                        and out_chn_counter = OUT_CHANNELS-1 then
                            out_chn_counter <= 0;
                            current_state <= IDLE;
                        end if;
                                                                        
                    when SET_PADDING =>
                        current_state <= READ_FIRST;
                        weight_map_index_start <= (out_chn_counter * WEIGHT_STRIDE);
                        first_index := pixel_counter*2 + (pixel_counter/OUT_SIZE) * IN_SIZE + POOLING_GROUP_OFFSET_TABLE(pooling_counter);
                        input_map_index <= first_index;
                        
                        -- set padding if index is on edge of input feature map
                        if first_index = 0 then -- top-left
                           padding_mask <= ('0','0','0',
                                            '0','1','1',
                                            '0','1','1');
                        elsif first_index = IN_SIZE-1 then -- top-right
                           padding_mask <= ('0', '0', '0',
                                            '1', '1', '0',
                                            '1', '1', '0');
                        elsif first_index = IN_SIZE_SQ - IN_SIZE then -- bot-left
                           padding_mask <= ('0', '1', '1',
                                            '0', '1', '1',
                                            '0', '0', '0');
                        elsif first_index = IN_SIZE_SQ-1 then --bot-right
                           padding_mask <= ('1', '1', '0',
                                            '1', '1', '0',
                                            '0', '0', '0');
                        elsif first_index < IN_SIZE then -- top-row
                           padding_mask <= ('0', '0', '0',
                                            '1', '1', '1',
                                            '1', '1', '1');
                        elsif first_index >= IN_SIZE_SQ - IN_SIZE then -- bot-row
                           padding_mask <= ('1', '1', '1',
                                            '1', '1', '1',
                                            '0', '0', '0');
                        elsif first_index mod IN_SIZE = 0 then -- left-col
                           padding_mask <= ('0', '1', '1',
                                            '0', '1', '1',
                                            '0', '1', '1');
                        elsif (first_index + 1) mod IN_SIZE = 0 then -- right-col
                           padding_mask <= ('1', '1', '0',
                                            '1', '1', '0',
                                            '1', '1', '0');
                        else -- not on edge
                           padding_mask <= ('1', '1', '1',
                                            '1', '1', '1',
                                            '1', '1', '1');
                        end if;
                    
                    when READ_FIRST => -- wait for mac to finish and start the capture process
                        current_state <= READING;
                        read_start <= '1';
                        
                        weight_map_index_offset <= weight_map_index_offset + 1;
                        input_map_index_offset  <= input_map_index_offset + 1;
                        
                        if weight_map_index_offset = KERNEL_SIZE_SQ*IN_CHANNELS-1 then
                            -- Last weight read
                            input_map_index_offset <= 0;
                            current_state <= WAIT_FOR_MAC;
                        end if;
                        
                    when READING =>
                        input_map_index_offset  <= input_map_index_offset + 1;
                        weight_map_index_offset <= weight_map_index_offset + 1;
                        
                        if weight_map_index_offset = KERNEL_SIZE_SQ*IN_CHANNELS-1 then
                            -- Last weight read
                            input_map_index_offset <= 0;
                            current_state <= WAIT_FOR_MAC; 
                        else
                            if input_map_index_offset = KERNEL_SIZE_SQ-1 then
                                -- go to next input channel
                                input_map_index_offset <= 0;
                                input_map_index  <= input_map_index + IN_SIZE_SQ;
                                current_state <= READ_FIRST;
                            end if;
                        end if;
                        
                    when WAIT_FOR_MAC =>
                        weight_map_index_offset <= 0;
                        if mac_finish = '1' then
                            if pooling_counter = 3 then
                                pooling_counter <= 0;
                                current_state <= INCREMENT;
                            else
                                pooling_counter <= pooling_counter + 1;
                                current_state <= SET_PADDING;
                            end if;
                        end if;
                        
                end case;
                
                
                
            end if;
        end if;
    end process control;
    
    RAM_capture : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                w_cnt <= 0;
                r_current_state <= IDLE; 
            else
                mac_ena <= '0';
                mac_clear <= '0';
                
                if padding_mask(w_cnt mod KERNEL_SIZE_SQ) = '1' then
                    mac_data_a <= brami_dout;
                else
                    mac_data_a <= (others => '0');
                end if;
                mac_data_b <= bramw_dout;
                
                case r_current_state is
                    when IDLE =>
                        if read_start = '1' then
                            mac_clear <= '1';
                            r_current_state <= INIT;
                        end if;
                        
                    when INIT =>
                        -- wait for one cycle for bram_dout to be ready
                        r_current_state <= CAPTURE;
                        w_cnt <= 0; 
                        
                    when CAPTURE =>
                        
                        
                        mac_ena <= '1';
                        w_cnt <= w_cnt + 1;
                        
                        if w_cnt = KERNEL_SIZE_SQ*IN_CHANNELS-1 then -- all weights read
                            r_current_state <= CAPTURE_BIAS;
                        end if;
                        
                    when CAPTURE_BIAS =>
                        -- Capture trailing bias for out feature
                        mac_ena <= '1';
                        mac_data_a <= x"3f800000"; -- 1
                        r_current_state <= IDLE;
                        
                end case;    
            end if;
        end if;
    end process RAM_capture;
    
    MAC_capture : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                -- Reset
                out_cnt <= 0;
                pool_cnt <= 0;
                done <= '0';
                mac_finish <= '0';
                pool_en <= '0';
            else
                -- Defaults
                done <= '0';
                mac_finish <= '0';
                pool_en <= '0';
                
                if out_cnt = OUT_CHANNELS*OUT_SIZE_SQ and pool_rdy = '1' then
                    done <= '1';
                elsif mac_ready = '1' then
                    mac_finish <= '1';
                    mac_results(pool_cnt) <= mac_sum_out;
                    
                    if pool_cnt = 3 then
                        pool_cnt <= 0;
                        pool_en <= '1';
                        out_cnt <= out_cnt + 1;
                    else
                        pool_cnt <= pool_cnt + 1;
                    end if;
                end if;
            
                bramo_addr <= std_logic_vector(TO_UNSIGNED(out_cnt-1, OUTPUT_MAP_ADDR_SIZE));
                
            end if;
        end if;
    end process MAC_capture;
    
    bramo_wea <= "" & pool_rdy;
    bramo_din <= pool_result;
    
end behavioral;