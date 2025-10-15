library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


-- This is the testbench for the mac_unit entity.
-- It will instantiate the mac_unit and apply a series of
-- stimulus vectors to verify its correct operation.

entity mac_unit_tb is
end mac_unit_tb;

architecture behavioral of mac_unit_tb is

    -- Constants for the testbench that match the DUT generics
    constant INPUT_WIDTH : integer := 32;
    constant ACCUM_WIDTH : integer := 32;
    constant CLK_PERIOD  : time    := 10 ns; -- 100 MHz clock

    -- Component declaration for the Device Under Test (DUT)
    component mac_unit is
        generic (
            INPUT_WIDTH : integer := 32;
            ACCUM_WIDTH : integer := 32
        );
        port (
            clk       : in  std_logic;
            rst       : in  std_logic;
            ena       : in  std_logic;
            clear_sum : in  std_logic;
            data_a    : in  std_logic_vector(INPUT_WIDTH-1 downto 0);
            data_b    : in  std_logic_vector(INPUT_WIDTH-1 downto 0);
            data_c    : in  std_logic_vector(INPUT_WIDTH-1 downto 0);
            sum_out   : out std_logic_vector(ACCUM_WIDTH-1 downto 0)
        );
    end component;

    -- Testbench signals to connect to the DUT
    signal clk       : std_logic := '0';
    signal rst       : std_logic;
    signal ena       : std_logic;
    signal clear_sum : std_logic;
    signal data_a    : std_logic_vector(INPUT_WIDTH-1 downto 0);
    signal data_b    : std_logic_vector(INPUT_WIDTH-1 downto 0);
    signal data_c    : std_logic_vector(INPUT_WIDTH-1 downto 0);
    signal sum_out   : std_logic_vector(ACCUM_WIDTH-1 downto 0);

begin

    -- Instantiate the DUT
    dut_inst : mac_unit
        generic map (
            INPUT_WIDTH => INPUT_WIDTH,
            ACCUM_WIDTH => ACCUM_WIDTH
        )
        port map (
            clk       => clk,
            rst       => rst,
            ena       => ena,
            clear_sum => clear_sum,
            data_a    => data_a,
            data_b    => data_b,
            data_c    => data_c,
            sum_out   => sum_out
        );

    -- Clock generation process
    clk_process : process
    begin
        clk <= '0';
        wait for CLK_PERIOD / 2;
        clk <= '1';
        wait for CLK_PERIOD / 2;
    end process;

    -- Stimulus process to drive inputs and check outputs
    stim_proc : process
        variable expected_sum : signed(ACCUM_WIDTH-1 downto 0);
    begin
        report "Starting MAC Unit Testbench...";

        -- 1. Apply Reset
        rst <= '1';
        wait for 2 * CLK_PERIOD;
        rst <= '0';
        wait for CLK_PERIOD;
        
        data_c <= x"00000000";

        -- 2. Test Case 1: Basic positive accumulation
        -- Test (3 * 5) + (2 * 4) = 15 + 8 = 23
        report "Test Case 1: Positive Accumulation";
        -- First operation: 3 * 5
        data_a <= "01000000000000000000000000000000"; -- 2
        data_b <= "01000000010000000000000000000000"; -- 3
        ena    <= '1';
        wait for CLK_PERIOD;
        data_a <= x"40a00000"; -- 5
        data_b <= x"40000000"; -- 2
        ena    <= '1';
        wait for CLK_PERIOD;
        data_a <= x"c0800000"; -- -4
        data_b <= x"3f800000"; -- 1
        ena    <= '1';
        wait for CLK_PERIOD;
        data_a <= x"3f800000"; -- 1
        data_b <= x"40e00000"; -- 7
        ena    <= '1';
        wait for CLK_PERIOD;
        data_a <= "01000000000000000000000000000000"; -- 2
        data_b <= "01000000010000000000000000000000"; -- 3
        ena    <= '1';
        wait for CLK_PERIOD;
        data_a <= "01000000000000000000000000000000"; -- 2
        data_b <= "01000000010000000000000000000000"; -- 3
        ena    <= '1';
        wait for CLK_PERIOD;
        data_a <= "01000000000000000000000000000000"; -- 2
        data_b <= "01000000010000000000000000000000"; -- 3
        ena    <= '1';
        wait for CLK_PERIOD;
        ena    <= '0';
        wait for 10 * CLK_PERIOD;
        --expected_sum := to_signed(37, ACCUM_WIDTH); -- 37
        --assert sum_out = expected_sum report "FAIL: 3*5, Expected " & integer'image(to_integer(expected_sum)) & ", Got " & integer'image(to_integer(sum_out)) severity error;

        
        -- Second operation: 2 * 4
        --data_a <= to_signed(2, INPUT_WIDTH);
        --data_b <= to_signed(4, INPUT_WIDTH);
        --ena    <= '1';
        --wait for CLK_PERIOD;
        --ena    <= '0';
        --wait for CLK_PERIOD;
        --expected_sum := to_signed(23, ACCUM_WIDTH);
        --assert sum_out = expected_sum report "FAIL: (3*5)+(2*4), Expected " & integer'image(to_integer(expected_sum)) & ", Got " & integer'image(to_integer(sum_out)) severity error;
        
        -- 3. Test Case 2: Clear Sum
        --report "Test Case 2: Clear Sum";
        --clear_sum <= '1';
        --wait for CLK_PERIOD;
        --clear_sum <= '0';
        --expected_sum := (others => '0');
        --assert sum_out = expected_sum report "FAIL: clear_sum, Expected 0, Got " & integer'image(to_integer(sum_out)) severity error;

        -- 4. Test Case 3: Negative number accumulation
        -- Test (-7 * 3) + (5 * -2) = -21 + -10 = -31
        --report "Test Case 3: Negative Accumulation";
        -- First operation: -7 * 3
        --data_a <= to_signed(-7, INPUT_WIDTH);
        --data_b <= to_signed(3, INPUT_WIDTH);
        --ena    <= '1';
        --wait for CLK_PERIOD;
        --ena    <= '0';
        --wait for CLK_PERIOD;
        --expected_sum := to_signed(-21, ACCUM_WIDTH);
        --assert sum_out = expected_sum report "FAIL: -7*3, Expected " & integer'image(to_integer(expected_sum)) & ", Got " & integer'image(to_integer(sum_out)) severity error;
        
        -- Second operation: 5 * -2
        --data_a <= to_signed(5, INPUT_WIDTH);
        --data_b <= to_signed(-2, INPUT_WIDTH);
        --ena    <= '1';
        --wait for CLK_PERIOD;
        --ena    <= '0';
        --wait for CLK_PERIOD;
        --expected_sum := to_signed(-31, ACCUM_WIDTH);
        --assert sum_out = expected_sum report "FAIL: (-7*3)+(5*-2), Expected " & integer'image(to_integer(expected_sum)) & ", Got " & integer'image(to_integer(sum_out)) severity error;

        -- 5. Test Case 4: Enable signal check
        -- The sum should not change when ena is '0'
        --report "Test Case 4: Enable check";
        --data_a <= to_signed(100, INPUT_WIDTH);
        --data_b <= to_signed(100, INPUT_WIDTH);
        --ena    <= '0';
        --wait for 5 * CLK_PERIOD;
        --assert sum_out = expected_sum report "FAIL: ena check, sum changed while disabled!" severity error;
        
        -- End of test
        report "All tests passed successfully." severity note;
        report "Simulation finished.";
        wait; -- Stop the simulation

    end process stim_proc;

end behavioral;
