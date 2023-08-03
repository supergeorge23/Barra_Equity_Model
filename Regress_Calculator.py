import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.api as sm
from my.data import basic_func
from Factor_Calculator import get_trade_date_rank, get_trade_date_from_rank
# Load the data
all_stock_data_np = np.load('all_stock_data_np.npy', allow_pickle=True)
compute_factor_matrix_normalized_20230512 = np.load('compute_factor_matrix_normalized_20230512.npy', allow_pickle=True)
industry_codes = np.load('all_stock_unique_industry_codes.npy', allow_pickle=True)
print("All Data have been successfully loaded!")


def add_stock_code_column(normalized_data, all_stock_data):
    # Get unique stock codes
    unique_items = np.unique(all_stock_data[:, 1])
    # Transform the normalized data into a pandas DataFrame
    df = pd.DataFrame(normalized_data, columns=['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP', 'DASTD',
                                                'CMRA', 'HSIGMA', 'EGRO', 'SGRO', 'BTOP', 'MLEV', 'DTOA',
                                                'BLEV', 'STOM', 'STOQ', 'STOA'])
    # Insert the 'stock_code' column at the start of the dataframe
    df.insert(0, 'stock_code', unique_items)
    return df
def get_pct_changes(date_str, unique_stock_codes):
    df_stock = basic_func.get_sqlserver(f"select * from AShareEODPrices where trade_dt='{date_str}'", "wind")
    stock_return = df_stock[['S_DQ_PCTCHANGE', 'S_INFO_WINDCODE']]
    stock_pct_changes = get_stock_pct_change(unique_stock_codes, stock_return)
    return stock_pct_changes
def get_stock_pct_change(unique_stock_codes, stock_return_df):
    # Initialize an array with zeros
    stock_pct_changes = np.zeros(len(unique_stock_codes))
    # Create a dictionary from the dataframe for faster lookup
    stock_return_dict = dict(zip(stock_return_df['S_INFO_WINDCODE'], stock_return_df['S_DQ_PCTCHANGE']))
    # Iterate through unique_stock_codes
    for i, stock_code in enumerate(unique_stock_codes):
        # If stock code is in the dataframe, update the corresponding index in stock_pct_changes
        if stock_code in stock_return_dict:
            stock_pct_changes[i] = stock_return_dict[stock_code]
    return stock_pct_changes

def parameter_estimation_WLS(trade_date_rank, industry_codes, all_stock_data_np, normalized_results_df, stock_returns):
    risk_factor_names = ['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP', 'DASTD', 'CMRA', 'HSIGMA',
                         'EGRO', 'SGRO', 'BTOP', 'MLEV', 'DTOA', 'BLEV', 'STOM', 'STOQ', 'STOA']
    industry_factor_names = ['Ind_' + str(i) for i in range(1201, 1220)]
    all_factor_names = risk_factor_names + industry_factor_names
    factor_loadings = []
    for _, stock in normalized_results_df.iterrows():
        # Extract factor loadings
        factors = [stock[risk_factor] if np.isfinite(stock[risk_factor]) else 0 for risk_factor in risk_factor_names]
        # Construct industry dummy variables
        industry_code_for_stock = industry_codes[industry_codes[:, 0] == stock['stock_code'], 1]
        if industry_code_for_stock.size == 0:
            industry_code_for_stock = np.array(['1220'])  # fallback if industry code not found
        industry_dummy_variables = [1 if 'Ind_' + industry_code == industry_code_for_stock else 0 for industry_code in
                                    industry_factor_names]
        # Add factor loadings and dummy variables to matrix
        factor_loadings.append(factors + industry_dummy_variables)
    factor_loadings = np.array(factor_loadings)
    # print("NaN values in factor_loadings:", np.isnan(factor_loadings).any())
    # print("NaN values in stock_returns:", np.isnan(stock_returns).any())
    # Remove rows with NaN values
    mask = ~np.isnan(factor_loadings).any(axis=1)
    factor_loadings_clean = factor_loadings[mask]
    stock_returns_clean = stock_returns[mask]
    # Solve for coefficients using numpy.linalg.lstsq
    if factor_loadings_clean.shape[0] > 0:
        coefficients, residuals, rank, singular_values = np.linalg.lstsq(factor_loadings_clean,
                                                                         stock_returns_clean,
                                                                         rcond=None)
    else:
        coefficients = np.zeros_like(all_factor_names, dtype=float)
        residuals = np.array([0])
    # Create a dictionary that pairs each coefficient with its corresponding factor name
    coefficient_dict = dict(zip(all_factor_names, coefficients))
    # Convert the dictionary to a pandas DataFrame for a nice tabular display
    coefficient_df = pd.DataFrame(list(coefficient_dict.items()), columns=['Factor', 'Coefficient'])
    # Calculate the residuals
    residuals = stock_returns_clean - np.dot(factor_loadings_clean, coefficients)
    # Calculate the square root of the residuals
    # Since residuals can be negative, we use the absolute value to ensure that the square root is a real number.
    sqrt_residuals = np.sqrt(np.abs(residuals))
    return coefficient_df, sqrt_residuals


# if __name__ == "__main__":
#     trade_date_rank = 1228
#     date_str = '20230512'
#     unique_stock_codes = np.unique(all_stock_data_np[:, 1])
#     compute_factor_matrix_normalized_20230512 = add_stock_code_column(compute_factor_matrix_normalized_20230512,
#                                                                       all_stock_data_np)
#     # print(f"compute_factor_matrix_normalized_20230512 value is be like: {compute_factor_matrix_normalized_20230512}, "
#     #       f"and its length is: {len(compute_factor_matrix_normalized_20230512)}")
#
#     pct_changes_20230512 = get_pct_changes(date_str, unique_stock_codes)
#     stock_returns = pct_changes_20230512
#     coefficient_df, sqrt_residuals_sum = parameter_estimation_OLS(trade_date_rank,
#                                                                   industry_codes,
#                                                                   all_stock_data_np,
#                                                                   compute_factor_matrix_normalized_20230512,
#                                                                   stock_returns)
#     print(f"coefficient_df value is : {coefficient_df}")
#     np.save('coefficient_df_20230512.npy', coefficient_df)
#     print(f"sqrt_residuals_sum value is : {sqrt_residuals_sum}")
#     np.save('sqrt_residuals_sum_20230512.npy', sqrt_residuals_sum.sum)

if __name__ == "__main__":
    unique_stock_codes = np.unique(all_stock_data_np[:, 1])
    # Define the start and end rank for the last 252 trading dates
    start_rank = 1228 - 251
    end_rank = 1228
    full_sqrt_residuals_sum_list = []
    full_coefficient_df = []
    for trade_date_rank in range(start_rank, end_rank + 1):
        date_str = get_trade_date_from_rank(trade_date_rank)
        compute_factor_matrix_normalized_date = np.load(f'compute_factor_matrix_normalized_{date_str}', allow_pickle=True)
        compute_factor_matrix_normalized_date = add_stock_code_column(
            compute_factor_matrix_normalized_date, all_stock_data_np)
        pct_changes_date = get_pct_changes(date_str, unique_stock_codes)
        stock_returns = pct_changes_date
        coefficient_df, sqrt_residuals_sum = parameter_estimation_OLS(
            trade_date_rank, industry_codes, all_stock_data_np,
            compute_factor_matrix_normalized_date, stock_returns)
        print(f"coefficient_df value for {date_str} is : {coefficient_df}")
        np.save(f'coefficient_df_{date_str}.npy', coefficient_df)
        full_coefficient_df.append(coefficient_df)
        np.save('full_coefficient_df.npy', full_coefficient_df)
        print(f"sqrt_residuals value for {date_str} is : {sqrt_residuals_sum}")
        np.save(f'sqrt_residuals_{date_str}.npy', sqrt_residuals_sum)
        print(f"sqrt_residuals_sum value for {date_str} is : {sqrt_residuals_sum.sum}")
        np.save(f'sqrt_residuals_sum_{date_str}.npy', sqrt_residuals_sum.sum)
        full_sqrt_residuals_sum_list.append(sqrt_residuals_sum.sum)
        np.save('full_sqrt_residuals_sum_list.npy', full_sqrt_residuals_sum_list)

