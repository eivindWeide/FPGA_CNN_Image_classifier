library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;


entity fp_mul is
    Port (
        -- Control Signals
        CLK   : in  std_logic;
        RST   : in  std_logic; -- reset
        EN    : in  std_logic; -- Enable
        
        -- In
        A : in std_logic_vector(31 downto 0);
        B : in std_logic_vector(31 downto 0);
        
        -- Out
        ready  : out std_logic;
        Output : out std_logic_vector(31 downto 0)
    );
end fp_mul;

architecture Pipelined of fp_mul is
    
    -- Pipeline Enable Signals
    signal EN_S1 : std_logic;
    signal EN_S2 : std_logic;
    signal EN_S3 : std_logic;
    
    -- S1 -> S2 Registers (Pre-computation results needed for S2)
    signal s1_mantissa_l_full : unsigned(23 downto 0);
    signal s1_mantissa_r_full : unsigned(23 downto 0);
    signal s1_exp_sum         : unsigned(8 downto 0);
    signal s1_result_sign     : std_logic;
    signal s1_is_special      : std_logic;
    signal s1_special_result  : std_logic_vector(31 downto 0);
    
    -- S2 -> S3 Registers (Multiplication result needed for S3)
    signal s2_mant_product    : unsigned(47 downto 0);
    signal s2_exp_sum         : unsigned(8 downto 0);
    signal s2_result_sign     : std_logic;
    signal s2_is_special      : std_logic;
    signal s2_special_result  : std_logic_vector(31 downto 0);

begin

    -- Pipeline Enable Logic: Stage N enabled by previous stage's enable
    process(CLK, RST)
    begin
        if RST = '1' then
            EN_S1 <= '0';
            EN_S2 <= '0';
        elsif rising_edge(CLK) then
            EN_S1 <= EN;
            EN_S2 <= EN_S1;
        end if;
    end process;


    FP_MUL_PROC : process (CLK, RST)
        -- Variables for input extraction (only used within S1)
        variable SIGN_L : std_logic;
        variable SIGN_R : std_logic;
        variable EXPONENT_L : unsigned(7 downto 0);
        variable EXPONENT_R : unsigned(7 downto 0);
        variable MANTISSA_L : unsigned(22 downto 0);
        variable MANTISSA_R : unsigned(22 downto 0);
        variable mantissa_l_full : unsigned(23 downto 0);
        variable mantissa_r_full : unsigned(23 downto 0);
        
        -- Variables for S3
        variable exp_sum_temp : unsigned(8 downto 0);
        variable mant_product_temp : unsigned(47 downto 0);
        variable result_exp_biased : unsigned(8 downto 0);
        variable result_mant : unsigned(22 downto 0);
        variable result : std_logic_vector(31 downto 0);
        variable special_result_temp : std_logic_vector(31 downto 0);
        
        CONSTANT INF_EXP   : std_logic_vector(7 downto 0)  := (others => '1');
        CONSTANT ZERO_EXP  : std_logic_vector(7 downto 0)  := (others => '0');
        CONSTANT ZERO_MANT : std_logic_vector(22 downto 0) := (others => '0');

    begin
        if RST = '1' then
            Output <= (others => '0');
            ready  <= '0';
        elsif rising_edge(CLK) then
            -- Default ready to 0 unless set in the final stage
            ready <= '0';

            -------------------------------------------------
            -- STAGE 1: Extraction and Pre-Computation
            -------------------------------------------------
            if EN = '1' then
                -- 1. Extract components
                SIGN_L := A(31);
                SIGN_R := B(31);
                EXPONENT_L := unsigned(A(30 downto 23));
                EXPONENT_R := unsigned(B(30 downto 23));
                MANTISSA_L := unsigned(A(22 downto 0));
                MANTISSA_R := unsigned(B(22 downto 0));
                
                -- Initialize full mantissas (implicit '1' included)
                mantissa_l_full := "1" & MANTISSA_L;
                mantissa_r_full := "1" & MANTISSA_R;

                -- 2. Pre-calculate normal results (runs in parallel with special case checks)
                s1_result_sign <= SIGN_L xor SIGN_R;
                s1_exp_sum     <= ('0' & EXPONENT_L) + ('0' & EXPONENT_R);
                s1_mantissa_l_full <= mantissa_l_full;
                s1_mantissa_r_full <= mantissa_r_full;
                --s1_mant_product <= mantissa_l_full * mantissa_r_full;

                -- 3. Check Special Cases and determine result/flag
                s1_is_special <= '0';
                
                -- NaN Check (A or B is NaN)
                if (EXPONENT_L = 255 and MANTISSA_L /= 0) or (EXPONENT_R = 255 and MANTISSA_R /= 0) then
                    s1_is_special <= '1';
                    s1_special_result <= x"7fc00000"; -- NaN
                
                -- 0 * Inf Check (special NaN case)
                elsif ((EXPONENT_L = 0 and MANTISSA_L = 0) and (EXPONENT_R = 255)) or ((EXPONENT_R = 0 and MANTISSA_R = 0) and (EXPONENT_L = 255)) then
                    s1_is_special <= '1';
                    s1_special_result <= x"7fc00000"; -- NaN

                -- Inf Check (A or B is Inf, excluding 0 * Inf handled above)
                elsif (EXPONENT_L = 255) or (EXPONENT_R = 255) then
                    s1_is_special <= '1';
                    s1_special_result <= (SIGN_L xor SIGN_R) & INF_EXP & ZERO_MANT; -- Inf

                -- Zero Check (A or B is Zero)
                elsif (EXPONENT_L = 0 and MANTISSA_L = 0) or (EXPONENT_R = 0 and MANTISSA_R = 0) then
                    s1_is_special <= '1';
                    s1_special_result <= (SIGN_L xor SIGN_R) & ZERO_EXP & ZERO_MANT; -- Zero (signed)
                end if;
            end if;

            -------------------------------------------------
            -- STAGE 2: Mantissa Multiplication
            -------------------------------------------------
            if EN_S1 = '1' then
                -- Pass through control/sign signals
                s2_is_special      <= s1_is_special;
                s2_special_result  <= s1_special_result;
                s2_exp_sum         <= s1_exp_sum;
                s2_result_sign     <= s1_result_sign;

                -- Only perform multiplication if not a special case
                if s1_is_special = '0' then
                    -- Multiply the full 24-bit mantissas
                    s2_mant_product <= s1_mantissa_l_full * s1_mantissa_r_full;
                else
                    s2_mant_product <= (others => '0');
                end if;
            end if;

            -------------------------------------------------
            -- STAGE 3: Normalization and Reconstruction
            -------------------------------------------------
            if EN_S2 = '1' then
                
                if s2_is_special = '1' then
                    -- Output the special case result
                    result := s2_special_result;
                else
                    -- 1. Check for overflow (product >= 2.0, bit 47 = '1')
                    if s2_mant_product(47) = '1' then
                        mant_product_temp := s2_mant_product srl 1;
                        exp_sum_temp := s2_exp_sum + 1;
                    else
                        -- Normal computation
                        mant_product_temp := s2_mant_product;
                        exp_sum_temp := s2_exp_sum;
                    end if;
                    
                    -- 2. Re-bias exponent
                    result_exp_biased := unsigned(to_signed(to_integer(exp_sum_temp), 9) - 127);
                    
                    -- 3. Truncate mantissa
                    -- Note: Truncation, not IEEE-754 rounding.
                    result_mant := mant_product_temp(45 downto 23);
                    
                    -- 4. Handle final overflow/underflow
                    if result_exp_biased >= 255 then 
                        -- Exponent overflow, result is infinity 
                        result := s2_result_sign & INF_EXP & ZERO_MANT; 
                        result(30 downto 23) := (others => '1');
                        result(31) := s2_result_sign;

                    elsif result_exp_biased <= 0 then
                        -- Exponent underflow, result is zero
                        -- Note: No handling of de-normalized numbers
                        result := (others => '0');
                        result(31) := s2_result_sign;
                    
                    else 
                        -- Reconstruct the final float
                        result := s2_result_sign & std_logic_vector(result_exp_biased(7 downto 0)) & std_logic_vector(result_mant);
                    end if;
                end if;

                -- Assign the calculated result to the output port
                ready <= '1';
                Output <= result;
            end if;
        end if;
    end process FP_MUL_PROC;

end Pipelined;