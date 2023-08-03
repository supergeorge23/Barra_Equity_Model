import pandas as pd
import numpy as np

np.random.seed(0)
# random_numbers = np.random.rand(18)
# random_numbers = random_numbers.T
# print(f"random_numbers value is: {random_numbers}, and its length is: {len(random_numbers)}")

# coefficient_df_1 = np.random.rand(1, 18)
# coefficient_df_2 = np.random.rand(1, 18)
# coefficient_df_3 = np.random.rand(1, 18)
# coefficient_df_4 = np.random.rand(1, 18)
# coefficient_df_5 = np.random.rand(1, 18)
# coefficient_df_6 = np.random.rand(1, 18)
# coefficient_df_7 = np.random.rand(1, 18)
# coefficient_df_8 = np.random.rand(1, 18)
# coefficient_df_9 = np.random.rand(1, 18)
# coefficient_df_10 = np.random.rand(1, 18)

def add_stock_code_column(coefficient_df):
    column_names = ['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP', 'DASTD', 'CMRA', 'HSIGMA', 'EGRO', 'SGRO',
                    'BTOP', 'MLEV', 'DTOA', 'BLEV', 'STOM', 'STOQ', 'STOA']
    # Convert numpy array to pandas DataFrame
    df = pd.DataFrame(data=coefficient_df, columns=column_names)
    return df
def add_trade_date_rank(df):
    df['listed_date_rank'] = np.arange(len(df), 0, -1)
    return df
def process_and_calculate_covariance(df, listed_date_rank, length_of_time, half_life_parameter, lambda_):
    def add_trade_date_rank(df):
        df['trade_date_rank'] = np.arange(len(df), 0, -1)
        return df
    def calculate_factor_return_covariance(df, t, h, r, lambda_):
        factors = ['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP', 'DASTD', 'CMRA', 'HSIGMA', 'EGRO', 'SGRO',
                   'BTOP', 'MLEV', 'DTOA', 'BLEV', 'STOM', 'STOQ', 'STOA']
        Fd = pd.DataFrame(index=factors, columns=factors)
        weights = [lambda_**(t-s) for s in range(t-h, t+1)]
        weights = [weight / sum(weights) for weight in weights]
        for factor_k in factors:
            for factor_k_prime in factors:
                sum_product = 0
                for s in range(t-h, t+1):
                    fk_s = df.loc[df['trade_date_rank'] == s, factor_k].values[0]
                    fk_prime_s = df.loc[df['trade_date_rank'] == s, factor_k_prime].values[0]
                    fk_bar = df[factor_k].mean()
                    fk_prime_bar = df[factor_k_prime].mean()
                    sum_product += weights[s - (t-h)] * (fk_s - fk_bar) * (fk_prime_s - fk_prime_bar)
                Fd.loc[factor_k, factor_k_prime] = sum_product
        return Fd
    df = add_trade_date_rank(df)
    Fd = calculate_factor_return_covariance(df, listed_date_rank, length_of_time, half_life_parameter, lambda_)
    # print(f"Fd value on date {listed_date_rank} is: \n {Fd} \n, and its length is: \n {len(Fd)}")
    return Fd
def calculate_factor_cross_sectional_deviation(df, t):
    factors = ['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP', 'DASTD', 'CMRA', 'HSIGMA', 'EGRO', 'SGRO',
               'BTOP', 'MLEV', 'DTOA', 'BLEV', 'STOM', 'STOQ', 'STOA']
    K = len(factors)  # number of factors
    sum_squared = 0
    # Iterate over all factors
    for factor in factors:
        # Get actual factor return and predicted volatility
        fk_t = df.loc[df['trade_date_rank'] == t, factor].values[0]
        # assuming volatilities are stored in the same dataframe
        sigma_kt = df.loc[df['trade_date_rank'] == t - 1, factor].values[0]
        # Add squared ratio to sum
        sum_squared += (fk_t / sigma_kt) ** 2
    # Calculate and return BtF
    BtF = np.sqrt(sum_squared / K)
    return BtF
def calculate_lambda_F(df, h, lambda_):
    # Calculate the time series of BtF values
    BtF_series = [calculate_factor_cross_sectional_deviation(df, t) for t in range(2, len(df) + 1)]  # range starts from 2
    BtF_series.reverse()
    # Calculate the weights
    weights = [lambda_**(t) for t in range(h)]
    weights = [weight / sum(weights) for weight in weights]
    # Calculate lambda_F
    lambda_F = np.sqrt(sum((BtF ** 2) * weight for BtF, weight in zip(BtF_series, weights)))
    return lambda_F
def adjust_Fd(Fd, lambda_F):
    return lambda_F ** 2 * Fd
def convert_to_monthly(Fd_tilde, C_NW):
    F_monthly = C_NW * Fd_tilde
    return F_monthly

all_coefficients = np.load('full_coefficient_df.npy', allow_pickle=True)
# Apply the function to all coefficient arrays and concatenate the results
combined_df = pd.concat([add_stock_code_column(coefficients) for coefficients in all_coefficients], ignore_index=True)
# print(f"combined_df value is \n {combined_df}")
combined_df = add_trade_date_rank(combined_df)
# Usage: need changes before use
listed_date_rank = 28
length_of_time = 3  # h value
half_life_parameter = 2
lambda_ = 0.94
df_28 = process_and_calculate_covariance(combined_df,
                                         listed_date_rank,
                                         length_of_time,
                                         half_life_parameter,
                                         lambda_)
# print(f"df value is be like: {df}")
BtF_28 = calculate_factor_cross_sectional_deviation(combined_df, listed_date_rank)
# print(f"BtF value on date 28 is: {BtF_28}")

lambda_F = calculate_lambda_F(combined_df, length_of_time, lambda_)

adjusted_Fd = adjust_Fd(df_28, lambda_F)
# print(f"Adjusted Fd is: \n {adjusted_Fd}")

C_NW = 0.94  # Example constant. Replace with actual Newey-West adjusted constant
F_monthly = convert_to_monthly(adjusted_Fd, C_NW)
print(f"Monthly frequency factor return covariance matrix: \n {F_monthly}")







