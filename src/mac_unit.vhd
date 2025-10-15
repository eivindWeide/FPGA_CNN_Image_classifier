library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- This entity performs a multiply-accumulate operation.
-- On each enabled clock cycle, it multiplies data_a and data_b,
-- and adds the result to an internal accumulator register (sum).
-- The sum can be reset to zero synchronously with the clear_sum signal.

entity mac_unit is
    generic (
        -- Width of the input operands
        INPUT_WIDTH : integer := 32
    );
    port (
        -- System signals
        clk       : in  std_logic;
        rst       : in  std_logic; -- Synchronous reset
        ena       : in  std_logic; -- Operation enable

        -- Control signal
        clear_sum : in  std_logic; -- Clears the internal sum to 0

        -- Data inputs (using signed type for arithmetic)
        data_a    : in  std_logic_vector(INPUT_WIDTH-1 downto 0);
        data_b    : in  std_logic_vector(INPUT_WIDTH-1 downto 0);

        -- Data output
        ready     : out std_logic;
        sum_out   : out std_logic_vector(INPUT_WIDTH-1 downto 0)
    );
end mac_unit;

architecture rtl of mac_unit is
	
	--  registers
	signal sum, product : std_logic_vector(31 downto 0);
	signal mul_a, mul_b, add_a, add_b : std_logic_vector(31 downto 0);
	
	
	
	signal total_sum : std_logic_vector(31 downto 0);
	
	signal mul_ready, add_ready, mul_ena, add_ena : std_logic;
    

begin

    mul : entity work.fp_mul
    port map(
        clk   => clk,
        rst   => rst,
        en    => mul_ena,
        A     => mul_a,
        B     => mul_b,
        ready => mul_ready,
        Output => product
        
    );
    
    add : entity work.fp_add
    port map(
        clk   => clk,
        rst   => rst,
        en    => add_ena,
        A     => add_a,
        B     => add_b,
        ready => add_ready,
        Output => sum
    );

    -- This process describes the sequential logic for the MAC operation.
    -- It is sensitive to the clock edge.
    mac_process : process(clk)
        -- Variable to count how many operations are currently pipelined.
        variable cnt : integer range -1 to 3 := 0;
    begin
        if rising_edge(clk) then
            
            ready <= '0';
            mul_a <= (others => '0');
            mul_b <= (others => '0');
            add_a <= (others => '0');
            add_b <= (others => '0');
            add_ena <= '0';
            mul_ena <= '0';
            
            
            -- Priority 1: System reset
            if rst = '1' then
                cnt := 0;
                total_sum <= (others => '0');

            -- Priority 2: Clear the sum on command
            elsif clear_sum = '1' then
                total_sum <= (others => '0');
                cnt := 0;

            -- Priority 3: Perform the MAC operation if enabled
            else
                if ena = '1' then
                    mul_a <= data_a;
                    mul_b <= data_b;
                    mul_ena <= '1';
                end if;
                
                if mul_ready = '1' and add_ready = '0' then
                    if total_sum = x"00000000" then
                        total_sum <= product; 
                    else
                        add_ena <= '1';
                        add_a <= product;
                        add_b <= total_sum;
                        total_sum <= x"00000000";
                        cnt := cnt + 1;
                    end if;
                
                elsif mul_ready = '1' and add_ready = '1' then
                    add_ena <= '1';
                    add_a <= product;
                    add_b <= sum;
                
                elsif mul_ready = '0' and add_ready = '1' then
                    if total_sum = x"00000000" then
                        total_sum <= sum;
                        cnt := cnt - 1;
                        if cnt = 0 then
                            ready <= '1';
                        end if;
                    else
                        add_ena <= '1';
                        add_a <= total_sum;
                        add_b <= sum;
                        total_sum <= x"00000000";
                    end if;
                end if;
            end if;
        end if;
    end process mac_process;
    
    -- Concurrently assign the internal sum register to the output port
    sum_out <= total_sum;

end rtl;
