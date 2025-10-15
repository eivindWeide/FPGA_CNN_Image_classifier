library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity feature_block_pass_tb is
end feature_block_pass_tb;

architecture behavioral of feature_block_pass_tb is

    -- Component Declaration for the Design Under Test (DUT)
    component feature_block_pass is
        generic (
            IN_CHANNELS     : integer := 3;
            OUT_CHANNELS    : integer := 32;
            IN_SIZE         : integer := 32;
            OUT_SIZE        : integer := 16;
            DATA_WIDTH      : integer := 32;
            KERNEL_SIZE     : integer := 9;
            IN_SIZE_SQ      : integer := 1024;
            RAM1_SIZE       : integer := 3072;
            RAM2_SIZE       : integer := 896;
            RAM3_SIZE       : integer := 8192
        );
        port (
            clk        : in  std_logic;
            rst        : in  std_logic;
            start      : in  std_logic;
            done       : out std_logic;
            bram1_dout : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            bram2_dout : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            bram1_addr : out std_logic_vector(RAM1_SIZE-1 downto 0);
            bram2_addr : out std_logic_vector(RAM2_SIZE-1 downto 0);
            bram3_we   : out std_logic;
            bram3_addr : out std_logic_vector(RAM3_SIZE-1 downto 0);
            bram3_din  : out std_logic_vector(DATA_WIDTH-1 downto 0)
        );
    end component;

    -- Component Declaration for the BRAM (used for BRAM3)
    component bram is
        generic (
            DATA_WIDTH : integer := 32;
            ADDR_WIDTH : integer := 5
        );
        port (
            clk  : in  std_logic;
            we   : in  std_logic;
            addr : in  std_logic_vector(ADDR_WIDTH-1 downto 0);
            din  : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            dout : out std_logic_vector(DATA_WIDTH-1 downto 0)
        );
    end component;

    -- Constants
    constant C_DATA_WIDTH   : integer := 32;
    constant C_RAM1_ADDR_WIDTH : integer := 12; -- Corresponds to RAM1_SIZE = 3072, so 2^12=4096 is sufficient
    constant C_RAM2_ADDR_WIDTH : integer := 10; -- Corresponds to RAM2_SIZE = 896, so 2^10=1024 is sufficient
    constant C_RAM3_ADDR_WIDTH : integer := 13; -- Corresponds to RAM3_SIZE = 8192, so 2^13=8192 is sufficient
    constant C_CLK_PERIOD   : time    := 10 ns;

    -- Signals
    signal s_clk        : std_logic := '0';
    signal s_rst        : std_logic;
    signal s_start      : std_logic;
    signal s_done       : std_logic;
    signal s_bram1_dout : std_logic_vector(C_DATA_WIDTH-1 downto 0);
    signal s_bram2_dout : std_logic_vector(C_DATA_WIDTH-1 downto 0);
    signal s_bram1_addr : std_logic_vector(3071 downto 0);
    signal s_bram2_addr : std_logic_vector(895 downto 0);
    signal s_bram3_we   : std_logic;
    signal s_bram3_addr : std_logic_vector(8191 downto 0);
    signal s_bram3_din  : std_logic_vector(C_DATA_WIDTH-1 downto 0);
    signal s_bram3_dout : std_logic_vector(C_DATA_WIDTH-1 downto 0); -- For reading from BRAM3 to verify

    -- BRAM model signals for BRAM1 and BRAM2
    type ram_array_type is array (natural range <>) of std_logic_vector(C_DATA_WIDTH-1 downto 0);
    signal ram1 : ram_array_type(0 to 2**C_RAM1_ADDR_WIDTH-1);
    signal ram2 : ram_array_type(0 to 2**C_RAM2_ADDR_WIDTH-1);


begin

    -- Instantiate the Design Under Test (DUT)
    dut_inst : feature_block_pass
        port map(
            clk        => s_clk,
            rst        => s_rst,
            start      => s_start,
            done       => s_done,
            bram1_dout => s_bram1_dout,
            bram2_dout => s_bram2_dout,
            bram1_addr => s_bram1_addr,
            bram2_addr => s_bram2_addr,
            bram3_we   => s_bram3_we,
            bram3_addr => s_bram3_addr,
            bram3_din  => s_bram3_din
        );

    -- Model BRAM1 and BRAM2 read ports
    bram1_model_proc : process(s_clk)
    begin
        if rising_edge(s_clk) then
            -- Model 1 cycle read latency, similar to the BRAM component
            s_bram1_dout <= ram1(to_integer(unsigned(s_bram1_addr(C_RAM1_ADDR_WIDTH-1 downto 0))));
        end if;
    end process;

    bram2_model_proc : process(s_clk)
    begin
        if rising_edge(s_clk) then
            s_bram2_dout <= ram2(to_integer(unsigned(s_bram2_addr(C_RAM2_ADDR_WIDTH-1 downto 0))));
        end if;
    end process;

    -- Instantiate BRAM3 (since it's written to by the DUT)
    bram3_inst : bram
        generic map (
            DATA_WIDTH => C_DATA_WIDTH,
            ADDR_WIDTH => C_RAM3_ADDR_WIDTH
        )
        port map (
            clk  => s_clk,
            we   => s_bram3_we,
            addr => s_bram3_addr(C_RAM3_ADDR_WIDTH-1 downto 0),
            din  => s_bram3_din,
            dout => s_bram3_dout
        );

    -- Clock process
    clk_process : process
    begin
        s_clk <= '0';
        wait for C_CLK_PERIOD / 2;
        s_clk <= '1';
        wait for C_CLK_PERIOD / 2;
    end process;

    -- Stimulus process
    stimulus_process : process
    begin
        -- Initialize BRAM1 and BRAM2 arrays
        report "Initializing BRAM1 and BRAM2 with test values." severity note;
        for i in 0 to 63 loop
            -- Fill the first 64 words with some data
            ram1(i) <= std_logic_vector(to_signed(100 - i, C_DATA_WIDTH)); -- BRAM1 gets 1, 2, 3...
            ram2(i) <= std_logic_vector(to_signed(100 - i, C_DATA_WIDTH)); -- BRAM2 gets 100, 99, 98...
        end loop;
        -- It's good practice to initialize the rest to zero
        for i in 64 to ram1'length-1 loop ram1(i) <= (others => '0'); end loop;
        for i in 64 to ram2'length-1 loop ram2(i) <= (others => '0'); end loop;

        -- Initialization and Reset
        s_start <= '0';
        s_rst <= '1';
        wait for C_CLK_PERIOD * 2;
        s_rst <= '0';
        wait for C_CLK_PERIOD;

        -- Start the process
        report "Starting the feature_block_pass operation." severity note;
        s_start <= '1';
        wait for C_CLK_PERIOD;
        s_start <= '0';

        -- Wait for completion
        wait until s_done = '1';
        report "Operation finished. 'done' signal received." severity note;

        -- Add checks here to read from BRAM3 and verify the results if desired

        report "Testbench finished." severity note;
        wait;
    end process;

end behavioral;

