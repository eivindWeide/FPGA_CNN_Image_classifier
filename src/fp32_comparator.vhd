library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fp32_comparator is
    port (
        -- Control Signals
        CLK   : in  std_logic;
        RST   : in  std_logic;
        EN    : in  std_logic;
        
        -- Data Inputs (Existing)
        A : in std_logic_vector(31 downto 0);
        B : in std_logic_vector(31 downto 0);
        C : in std_logic_vector(31 downto 0);
        D : in std_logic_vector(31 downto 0);
        
        -- Data Outputs
        Largest_Output : out std_logic_vector(31 downto 0);
        ready : out std_logic
    );
end entity fp32_comparator;

architecture Synchronous of fp32_comparator is

    function max_fp32 (
        L : std_logic_vector(31 downto 0);
        R : std_logic_vector(31 downto 0)
    ) return std_logic_vector is
        -- Extract signs
        constant SIGN_L : std_logic := L(31);
        constant SIGN_R : std_logic := R(31);
        
        -- Convert to unsigned for bit-level comparison
        constant L_unsigned : unsigned(31 downto 0) := unsigned(L);
        constant R_unsigned : unsigned(31 downto 0) := unsigned(R);
        
        -- Result variable
        variable Result : std_logic_vector(31 downto 0);
    begin
        if L = R then
            Result := L; -- Equal, return either 
        elsif SIGN_L = '0' and SIGN_R = '1' then
            -- L is Positive, R is Negative: L is largest 
            Result := L;
        elsif SIGN_L = '1' and SIGN_R = '0' then
            -- L is Negative, R is Positive: R is largest
            Result := R;
        elsif SIGN_L = '0' and SIGN_R = '0' then
            -- Both Positive: Standard unsigned comparison (larger bit pattern is larger)
            if L_unsigned > R_unsigned then
                Result := L;
            else
                Result := R;
            end if;
        else 
            -- setting to 0 because there is supposed to be a ReLU layer after the convolution and this is
            -- the easiest way to implement it
            Result := x"00000000"; 
        end if;
        return Result;
    end function max_fp32;

    -- Signal declarations
    -- Combinational outputs of Stage 1 (always calculated)
    signal Max_AB_comb : std_logic_vector(31 downto 0);
    signal Max_CD_comb : std_logic_vector(31 downto 0);
    
    -- Pipeline Registers (Registers to hold Stage 1 results)
    signal Max_AB_reg : std_logic_vector(31 downto 0);
    signal Max_CD_reg : std_logic_vector(31 downto 0);
    signal EN_S2      : std_logic; -- Enable signal for the second stage

begin
    -- COMBINATIONAL STAGE 1: Calculate Max(A, B) and Max(C, D)
    -- This logic runs continuously and the results are registered in the next clock cycle.
    Max_AB_comb <= max_fp32(A, B);
    Max_CD_comb <= max_fp32(C, D);

    -- SYNCHRONOUS PROCESS: Registers the pipeline stages
    COMPARISON_PROC : process (CLK, RST)
    begin
        if RST = '1' then
            Max_AB_reg     <= (others => '0');
            Max_CD_reg     <= (others => '0');
            Largest_Output <= (others => '0');
            EN_S2          <= '0';
            ready          <= '0';
        elsif rising_edge(CLK) then
            ready <= '0'; -- Ready defaults low unless set
            
            -- STAGE 1: Input Registration
            if EN = '1' then
                -- Register the combinational result from the current cycle's inputs (A, B, C, D)
                Max_AB_reg <= Max_AB_comb; 
                Max_CD_reg <= Max_CD_comb;
                EN_S2      <= '1'; -- Enable Stage 2 in the next cycle
            else
                EN_S2 <= '0';
            end if;
            
            -- STAGE 2: Final Comparison and Output
            -- This stage uses the registered results from the previous clock cycle
            if EN_S2 = '1' then
                Largest_Output <= max_fp32(Max_AB_reg, Max_CD_reg);
                ready <= '1'; -- Final output is valid
            end if;
        end if;
    end process COMPARISON_PROC;

end architecture Synchronous;