library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity fp_add is
    Port (
        -- Control Signals
        CLK   : in  std_logic;
        RST   : in  std_logic; -- reset
        EN    : in  std_logic; -- Enable

        -- Data Inputs
        A : in std_logic_vector(31 downto 0);
        B : in std_logic_vector(31 downto 0);

        -- Data Output
        ready  : out std_logic;
        Output : out std_logic_vector(31 downto 0)
    );
end fp_add;

architecture Behavioral of fp_add is

    function clz25(v : unsigned(24 downto 0)) return natural is
    begin
        if v(24) = '1' then return 0;
        elsif v(23) = '1' then return 1;
        elsif v(22) = '1' then return 2;
        elsif v(21) = '1' then return 3;
        elsif v(20) = '1' then return 4;
        elsif v(19) = '1' then return 5;
        elsif v(18) = '1' then return 6;
        elsif v(17) = '1' then return 7;
        elsif v(16) = '1' then return 8;
        elsif v(15) = '1' then return 9;
        elsif v(14) = '1' then return 10;
        elsif v(13) = '1' then return 11;
        elsif v(12) = '1' then return 12;
        elsif v(11) = '1' then return 13;
        elsif v(10) = '1' then return 14;
        elsif v(9)  = '1' then return 15;
        elsif v(8)  = '1' then return 16;
        elsif v(7)  = '1' then return 17;
        elsif v(6)  = '1' then return 18;
        elsif v(5)  = '1' then return 19;
        elsif v(4)  = '1' then return 20;
        elsif v(3)  = '1' then return 21;
        elsif v(2)  = '1' then return 22;
        elsif v(1)  = '1' then return 23;
        elsif v(0)  = '1' then return 24;
        else return 25;
        end if;
    end function clz25;

    -- Pipeline Registers for Stage 1 -> Stage 2
    signal s1_mantissa_l_full : unsigned(23 downto 0);
    signal s1_mantissa_r_full : unsigned(23 downto 0);
    signal s1_exponent_l      : unsigned(7 downto 0);
    signal s1_exponent_r      : unsigned(7 downto 0);
    signal s1_sign_l          : std_logic;
    signal s1_sign_r          : std_logic;
    signal s1_A_is_special    : std_logic;
    signal s1_B_is_special    : std_logic;
    signal s1_special_result  : std_logic_vector(31 downto 0);
    
    -- Pipeline Registers for Stage 2 -> Stage 3
    signal s2_aligned_mant_l : unsigned(24 downto 0); -- Extend for the 0 prepended in S3
    signal s2_aligned_mant_r : unsigned(24 downto 0);
    signal s2_result_exp_biased : unsigned(8 downto 0);
    signal s2_sign_l         : std_logic;
    signal s2_sign_r         : std_logic;
    signal s2_special_flag   : std_logic;
    signal s2_special_result : std_logic_vector(31 downto 0);
    
    -- Pipeline Registers for Stage 3 -> Stage 4
    signal s3_result_mant    : unsigned(24 downto 0);
    signal s3_result_sign    : std_logic;
    signal s3_result_exp_biased : unsigned(8 downto 0);
    signal s3_special_flag   : std_logic;
    signal s3_special_result : std_logic_vector(31 downto 0);
    signal s3_was_addition : std_logic;
    signal s3_num_leading_zeros : natural range 0 to 25;
    
    -- Ready signals for each stage
    signal ready_s1 : std_logic;
    signal ready_s2 : std_logic;
    signal ready_s3 : std_logic;
    
    -- Pipeline Enable for each stage
    signal EN_S1 : std_logic;
    signal EN_S2 : std_logic;
    signal EN_S3 : std_logic;

begin

    -- Pipeline Enable Logic
    -- Simple pipeline: All stages are enabled when the initial EN is high.
    -- For a more complex design, you'd use a control unit to manage stalls.
    process(CLK, RST)
    begin
        if RST = '1' then
            EN_S1 <= '0';
            EN_S2 <= '0';
            EN_S3 <= '0';
        elsif rising_edge(CLK) then
            EN_S1 <= EN;
            EN_S2 <= EN_S1;
            EN_S3 <= EN_S2;
        end if;
    end process;


    FP_ADD_PROC : process (CLK, RST)
        -- Variables for S1
        variable mantissa_l_full : unsigned(23 downto 0);
        variable mantissa_r_full : unsigned(23 downto 0);
        variable EXPONENT_L : unsigned(7 downto 0);
        variable EXPONENT_R : unsigned(7 downto 0);
        variable MANTISSA_L : unsigned(22 downto 0);
        variable MANTISSA_R : unsigned(22 downto 0);
        variable SIGN_L     : std_logic;
        variable SIGN_R     : std_logic;
            
        -- Variables for S2
        variable exp_diff : unsigned(7 downto 0);
        variable mantissa_r_shifted : unsigned(24 downto 0);
        variable mantissa_l_shifted : unsigned(24 downto 0);
        
        -- Variables for S3
        variable result_mant : unsigned(24 downto 0);
        variable result_sign : std_logic;
        
        -- Variables for S4
        variable shift_count : natural range 0 to 24;
        variable effective_shift : natural range 0 to 24;
        variable result_exp_temp : unsigned(8 downto 0);
        variable result : std_logic_vector(31 downto 0);

    begin
        if RST = '1' then
            Output <= (others => '0');
            ready  <= '0';
        elsif rising_edge(CLK) then
            -- Default ready to 0 unless set in the final stage
            ready <= '0';

            -------------------------------------------------
            -- STAGE 1: Pre-Processing and Extraction
            -------------------------------------------------
            if EN = '1' then
                -- Extract components
                SIGN_L := A(31);
                SIGN_R := B(31);
                EXPONENT_L := unsigned(A(30 downto 23));
                EXPONENT_R := unsigned(B(30 downto 23));
                MANTISSA_L := unsigned(A(22 downto 0));
                MANTISSA_R := unsigned(B(22 downto 0));

                -- Initialize full mantissas with space for the implicit '1' 
                mantissa_l_full := '0' & MANTISSA_L;
                mantissa_r_full := '0' & MANTISSA_R;

                -- Add leading '1' to mantissa if the number is normalized
                if EXPONENT_L /= 0 then
                    mantissa_l_full(23) := '1';
                end if;
                if EXPONENT_R /= 0 then
                    mantissa_r_full(23) := '1';
                end if;

                -- Handle special cases (NaN, Infinity, Zero)
                s1_A_is_special <= '0';
                s1_B_is_special <= '0';

                if EXPONENT_L = 255 and MANTISSA_L /= 0 then      -- A is NaN
                    s1_A_is_special <= '1';
                    s1_special_result <= A;
                elsif EXPONENT_R = 255 and MANTISSA_R /= 0 then   -- B is NaN
                    s1_B_is_special <= '1';
                    s1_special_result <= B;
                elsif EXPONENT_L = 255 and EXPONENT_R = 255 and SIGN_L /= SIGN_R then -- Inf - Inf
                    s1_A_is_special <= '1'; -- Use A to signify a special case
                    s1_special_result <= x"7fc00000"; -- Quiet NaN
                elsif EXPONENT_L = 0 and MANTISSA_L = 0 then      -- 0 + B = B 
                    s1_A_is_special <= '1';
                    s1_special_result <= B;
                elsif EXPONENT_R = 0 and MANTISSA_R = 0 then      -- A + 0 = A 
                    s1_B_is_special <= '1';
                    s1_special_result <= A;
                end if;

                -- Pass data to Stage 2 registers
                s1_mantissa_l_full <= mantissa_l_full;
                s1_mantissa_r_full <= mantissa_r_full;
                s1_exponent_l      <= EXPONENT_L;
                s1_exponent_r      <= EXPONENT_R;
                s1_sign_l          <= SIGN_L;
                s1_sign_r          <= SIGN_R;

            end if;

            -------------------------------------------------
            -- STAGE 2: Exponent Alignment
            -------------------------------------------------
            if EN_S1 = '1' then
                s2_special_flag <= s1_A_is_special or s1_B_is_special;
                s2_special_result <= s1_special_result;

                -- Only perform alignment if not a special case
                if (s1_A_is_special = '0') and (s1_B_is_special = '0') then
                    -- Extend mantissas for alignment (24 bits -> 25 bits to capture shifts)
                    mantissa_l_shifted := ('0' & s1_mantissa_l_full);
                    mantissa_r_shifted := ('0' & s1_mantissa_r_full);

                    if s1_exponent_l > s1_exponent_r then
                        exp_diff := s1_exponent_l - s1_exponent_r;
                        mantissa_r_shifted := shift_right(mantissa_r_shifted, TO_INTEGER(exp_diff));
                        s2_result_exp_biased <= ('0' & s1_exponent_l);
                    elsif s1_exponent_l < s1_exponent_r then
                        exp_diff := s1_exponent_r - s1_exponent_l;
                        mantissa_l_shifted := shift_right(mantissa_l_shifted, TO_INTEGER(exp_diff));
                        s2_result_exp_biased <= ('0' & s1_exponent_r);
                    else -- Exponents are equal
                        s2_result_exp_biased <= ('0' & s1_exponent_l);
                    end if;

                    -- Pass data to Stage 3 registers
                    s2_aligned_mant_l <= mantissa_l_shifted;
                    s2_aligned_mant_r <= mantissa_r_shifted;
                    s2_sign_l <= s1_sign_l;
                    s2_sign_r <= s1_sign_r;
                else
                    -- Pass through defaults/placeholder for non-special case signals
                    s2_aligned_mant_l <= (others => '0');
                    s2_aligned_mant_r <= (others => '0');
                    s2_result_exp_biased <= (others => '0');
                    s2_sign_l <= '0';
                    s2_sign_r <= '0';
                end if;
            end if;

            -------------------------------------------------
            -- STAGE 3: Mantissa Addition/Subtraction
            -------------------------------------------------
            if EN_S2 = '1' then
                s3_special_flag <= s2_special_flag;
                s3_special_result <= s2_special_result;
                s3_result_exp_biased <= s2_result_exp_biased;

                

                -- record whether this was an addition (signs equal)
                s3_was_addition <= not (s2_sign_l xor s2_sign_r);

                -- Only perform calculation if not a special case
                if s2_special_flag = '0' then
                    if s2_sign_l = s2_sign_r then -- Addition
                        result_mant := s2_aligned_mant_l + s2_aligned_mant_r;
                        result_sign := s2_sign_l;
                    else -- Subtraction
                        if s2_aligned_mant_l >= s2_aligned_mant_r then
                            result_mant := s2_aligned_mant_l - s2_aligned_mant_r;
                            result_sign := s2_sign_l;
                        else
                            result_mant := s2_aligned_mant_r - s2_aligned_mant_l;
                            result_sign := s2_sign_r;
                        end if;
                    end if;
                    
                    s3_num_leading_zeros <= clz25(result_mant);
                    s3_result_mant <= result_mant;
                    s3_result_sign <= result_sign;
                else
                    s3_result_mant <= (others => '0');
                    s3_result_sign <= '0';
                    s3_num_leading_zeros <= 0;
                end if;
            end if;

            -------------------------------------------------
            -- STAGE 4: Normalization and Reconstruction
            -------------------------------------------------
            if EN_S3 = '1' then
                
                if s3_special_flag = '1' then
                    -- Output the special case result
                    result := s3_special_result;
                else
                    -- Normal computation branch
                    result_exp_temp := s3_result_exp_biased;
                    result_mant := s3_result_mant;
                    
                    -- Check for overflow from addition (result_mant(24) = '1')
                    if s3_was_addition = '1' and s3_result_mant(24) = '1' then
                        -- Addition caused an overflow, shift right and increment exponent
                        result_mant := shift_right(s3_result_mant, 1);
                        result_exp_temp := s3_result_exp_biased + 1;
                    else
                        -- Subtraction or no overflow, perform normalization shift
                        if s3_num_leading_zeros = 25 then -- Result is zero
                            shift_count := 0;
                            result_exp_temp := (others => '0');
                        else
                            shift_count := s3_num_leading_zeros - 1;
                        end if;

                        -- Apply shift and update exponent, checking for underflow
                        if shift_count > 0 then
                            if result_exp_temp <= shift_count then
                                effective_shift := to_integer(result_exp_temp) - 1;
                                if effective_shift > 0 then
                                     result_mant := shift_left(s3_result_mant, effective_shift);
                                end if;
                                result_exp_temp := (others => '0'); -- Exponent underflow to zero
                            else
                                result_mant := shift_left(s3_result_mant, shift_count);
                                result_exp_temp := result_exp_temp - shift_count;
                            end if;
                        end if;
                    end if;

                    -- Handle final overflow/underflow and reconstruct the float
                    if result_exp_temp >= 255 then
                        -- Exponent overflow, result is infinity
                        result := (others => '0');
                        result(31) := s3_result_sign;
                        result(30 downto 23) := (others => '1');
                    elsif result_exp_temp <= 0 then
                        -- Exponent underflow, result is zero/denormalized
                        result := (others => '0');
                        result(31) := s3_result_sign;
                        -- If result_exp_temp is 0, the number is already denormalized/zero
                        -- If the denormalized calculation is complex, it's often simplified to zero.
                    else
                        -- Reconstruct the final floating-point number
                        result := s3_result_sign & std_logic_vector(result_exp_temp(7 downto 0)) & std_logic_vector(result_mant(22 downto 0));
                    end if;
                end if;

                -- Assign the final result to the output port
                ready <= '1';
                Output <= result;
            end if;
        end if;
    end process FP_ADD_PROC;

end Behavioral;