import numpy as np
from my.data import basic_func
import pandas as pd
import time
from datetime import datetime, timedelta
import os
all_stock_data_np = np.load('all_stock_data_np.npy', allow_pickle=True)
all_index_data_np = np.load('all_index_data_np.npy', allow_pickle=True)
print("All data has been successfully loaded!")

#################     This is for public functions     #################
def calculate_residuals_numpy(X, Y):
    try:
        # Add a column of ones to X for the intercept term
        X = np.column_stack((np.ones(X.shape[0]), X))
        # Calculate the coefficient vector B
        B = np.linalg.inv(X.T @ X) @ (X.T @ Y)
        # Calculate and return the residuals
        residuals = Y - X @ B
        return residuals.sum()
    except Exception as e:
        # print(f"Error calculating residuals: {e}")
        return 0
def get_trade_date_from_rank(target_date_rank_0):
    # Convert the target_date_rank to an actual trading date
    target_date_0 = all_stock_data_np[all_stock_data_np[:, -1] == target_date_rank_0, 2][0]
    return target_date_0
def get_trade_date_rank(target_date_0):
    # Retrieve the 'trade_date_rank' for the target_date
    target_date_rank_0 = all_stock_data_np[all_stock_data_np[:, 2] == target_date_0, -1][0]
    return target_date_rank_0
def create_stock_data_matrix(sorted_data, target_trade_date, col_idx_list):
    # Identify unique stock codes
    unique_stock_codes = np.unique(sorted_data[:, 1])
    print("Unique stock codes have been collected!")
    # Initialize an empty list to hold the data for each stock
    stock_data_list = []
    # For each unique stock code, extract data from all specified columns and append to the list
    for stock_code in unique_stock_codes:
        stock_data = sorted_data[sorted_data[:, 1] == stock_code][:, col_idx_list]
        stock_data_list.append(stock_data)
    # Convert the list to a numpy array and transpose it
    # So that each row represents a unique stock's data
    stock_data_matrix = np.array(stock_data_list)
    # Save the data matrix to a numpy file, using the target_trade_date in the filename
    np.save(f'stock_data_matrix_{target_trade_date}.npy', stock_data_matrix)
    print(f"Stock data matrix for {target_trade_date} has been saved!")
    return stock_data_matrix
def sort_data(all_stock_data_np):
    # Create an order with 'S_INFO_WINDCODE' first and 'trade_date_rank' second
    order = np.lexsort((all_stock_data_np[:, -1], all_stock_data_np[:, 1]))
    # Apply the sort
    all_stock_data_np_sorted = all_stock_data_np[order]
    return all_stock_data_np_sorted
def split_data_by_stock_code(all_stock_data_np):
    # Split all_stock_data_np into sub-arrays for each unique 'S_INFO_WINDCODE'
    _, index = np.unique(all_stock_data_np[:, 1], return_index=True)
    split_arrays = np.split(all_stock_data_np, index[1:])
    return split_arrays
def retrieve_first_items(data_matrix, index):
    result = []
    for inner_matrix in data_matrix:
        # Take the first item from each row of the inner matrix and add it to the result
        first_items = inner_matrix[:, index]
        result.append(first_items)
    # Convert the result to a numpy array for consistency
    return np.array(result)
def prepare_market_data(market_data):
    # Ensure the market data is sorted by date
    market_data_sorted = market_data[market_data[:, -1].argsort()]
    # Filter for market data
    market_data_filtered = market_data_sorted[market_data_sorted[:, 1] == '000300.SH']
    # Filter the last 252 trading days for the market index
    last_252_days_market = market_data_filtered[-252:, 10].astype(float)
    # Compute market returns
    market_returns = last_252_days_market[1:] / last_252_days_market[:-1] - 1
    return market_returns
def get_stock_data_matrices(stock_data_matrix, indices):
    return [retrieve_first_items(stock_data_matrix, index) for index in indices]
def retrieve_AShareIncome_data(statement_type):
    # Fetch AShareIncome data for all stocks with the specific statement type
    df_AShareIncome = basic_func.get_sqlserver(
        f"select * from AShareIncome where STATEMENT_TYPE='{statement_type}'",
        "wind")
    # Filter out rows with 'A' in the stock code
    df_AShareIncome = df_AShareIncome[~df_AShareIncome['S_INFO_WINDCODE'].str.contains('A')]
    # Sort by 'S_INFO_WINDCODE'
    df_AShareIncome = df_AShareIncome.sort_values(by='S_INFO_WINDCODE')
    return df_AShareIncome
def create_AShareIncome_data_matrix(statement_type, target_trade_date, col_idx_list):
    # Retrieve and sort AShareIncome data
    df_AShareIncome = retrieve_AShareIncome_data(statement_type)
    # Convert the dataframe to a numpy array
    sorted_data = df_AShareIncome.to_numpy()
    # Create and save the data matrix
    stock_data_matrix = create_stock_data_matrix(sorted_data, target_trade_date, col_idx_list)
    return stock_data_matrix
def retrieve_AShareBalanceSheet_data(statement_type):
    # Fetch AShareBalanceSheet data for all stocks with the specific statement type
    df_AShareBalanceSheet = basic_func.get_sqlserver(
        f"select * from AShareBalanceSheet where STATEMENT_TYPE='{statement_type}'", "wind")
    # Filter out rows with 'A' in the stock code
    df_AShareBalanceSheet = df_AShareBalanceSheet[~df_AShareBalanceSheet['S_INFO_WINDCODE'].str.contains('A')]
    # Sort by 'S_INFO_WINDCODE'
    df_AShareBalanceSheet = df_AShareBalanceSheet.sort_values(by='S_INFO_WINDCODE')
    return df_AShareBalanceSheet
def get_last_values(input_array):
    # Retrieve the last item of each array and write them into a new list
    last_values = [arr[-1] for arr in input_array if len(arr) > 0]  # Ensure the sub-array is not empty
    # Convert the list to a numpy array
    last_values_array = np.array(last_values)
    # Convert nan values to 0
    last_values_array = np.nan_to_num(last_values_array)
    return last_values_array
#################        End of public functions       #################

def run_the_main(target_trade_date):
    #################       This is for risk factors       #################
    def calculate_NLSIZE_numpy(stock_data):
        try:
            # Ensure the market value data is numeric and does not contain zero or negative values
            market_value = stock_data[:, 4].astype(np.float64)  # Convert to float
            market_value[market_value <= 0] = np.finfo(
                np.float64).tiny  # Replace zero or negative values with the smallest positive number
            # Calculate the logarithmic market value
            log_market_value = np.log(market_value)
            # Remove or replace NaN and inf values
            log_market_value = log_market_value[~np.isinf(log_market_value)]  # Removes -inf, inf
            log_market_value = log_market_value[~np.isnan(log_market_value)]  # Removes NaN
            # Create the regressor which is the cube of the logarithmic market value
            cubic_log_market_value = log_market_value ** 3
            # Calculate and return the sum of residuals
            return calculate_residuals_numpy(cubic_log_market_value, log_market_value)
        except Exception as e:
            print(f"Error calculating NLSIZE for stock: {e}")
            return 0
    def calculate_all_betas_numpy(stock_data_matrix, market_data):
        # Ensure the market data is sorted by date
        market_data_sorted = market_data[market_data[:, -1].argsort()]
        # Filter for market data
        market_data_filtered = market_data_sorted[market_data_sorted[:, 1] == '000300.SH']
        # Filter the last 250 trading days for the market index
        last_250_days_market = market_data_filtered[-250:, 10].astype(float)
        # Compute market returns
        market_returns = last_250_days_market[1:] / last_250_days_market[:-1] - 1
        # Create weights with half-life of 60 days
        weights = 0.5 ** (np.arange(249) / 60)  # 249 returns based on 250 price observations
        betas = []
        for stock_prices in stock_data_matrix:
            # Get last 250 trading days for each stock
            last_250_days_stocks = stock_prices[-250:].astype(float)
            # Compute stock returns
            stock_returns = last_250_days_stocks[1:] / last_250_days_stocks[:-1] - 1
            # Only calculate beta if we have enough data
            if len(stock_returns) == len(market_returns):
                # Calculate weighted covariance between stock and market
                covariance = np.cov(stock_returns, market_returns, aweights=weights)
                # Calculate variance of market
                market_var = np.var(market_returns, ddof=1)
                # Calculate the beta coefficients for the stock (covariance / market variance)
                beta = covariance[0, 1] / market_var
                betas.append(beta)
            else:
                betas.append(np.nan)  # Append NaN for stocks with insufficient data
        return np.array(betas)
    def calculate_RSTR_vectorized(stock_data_sorted):
        T = 500
        L = 21
        # Convert the closing price data to float
        closing_prices = stock_data_sorted[:, 21].astype(float)
        # Initialize returns array
        returns = np.empty(closing_prices.shape)
        returns[0] = np.nan  # set the initial value
        # Calculate returns while handling zeros in the denominator
        mask = closing_prices[:-1] != 0  # create a mask for non-zero values
        returns[1:][mask] = np.log(closing_prices[1:][mask] / closing_prices[:-1][mask])
        returns[1:][~mask] = np.nan  # replace division by zero with nan
        # Create weights with half-life of 120 days
        weights = 0.5 ** (np.arange(T + L - 1, -1, -1) / 120)
        # Check if we have enough data
        if len(returns) < T:
            return np.nan
        # Calculate the weighted sum of returns over the last T days
        RSTR = np.nansum(returns[-T:] * weights[:T])
        return RSTR
    def calculate_LNCAP_vectorized(market_values_matrix):
        LNCAP_values = []
        for market_values in market_values_matrix:
            try:
                # Convert 'S_VAL_MV' to float type
                S_VAL_MV = market_values.astype(float)
                # Calculate LNCAP for all stocks
                LNCAP_value = np.log(S_VAL_MV)
                # Replace inf and -inf with nan
                LNCAP_value[np.isinf(LNCAP_value)] = np.nan
                LNCAP_values.append(LNCAP_value)
            except Exception as e:
                print(f"Error while processing: {e}")
                LNCAP_values.append(np.nan)
        return np.array(LNCAP_values)
    def calculate_ETOP_vectorized(S_VAL_PE_TTM_matrix):
        ETOP_values = []
        for S_VAL_PE_TTM in S_VAL_PE_TTM_matrix:
            try:
                # Convert 'S_VAL_PE_TTM' to float type
                S_VAL_PE_TTM = S_VAL_PE_TTM.astype(float)
                # Calculate ETOP for all stocks
                ETOP_value = 1 / S_VAL_PE_TTM
                # Replace inf and -inf with nan
                ETOP_value[np.isinf(ETOP_value)] = np.nan
                ETOP_values.append(ETOP_value)
            except Exception as e:
                print(f"Error while processing: {e}")
                ETOP_values.append(np.nan)
        return np.array(ETOP_values)
    def calculate_CETOP_vectorized(net_cash_flow_matrix, market_cap_matrix):
        CETOP_values = []
        for net_cash_flow, market_cap in zip(net_cash_flow_matrix, market_cap_matrix):
            try:
                # Convert 'NET_CASH_FLOWS_OPER_ACT_TTM' and 'S_VAL_MV' to float type
                net_cash_flow = net_cash_flow.astype(float)
                market_cap = market_cap.astype(float)
                # Calculate CETOP for all stocks
                CETOP_value = net_cash_flow / market_cap
                # Replace inf and -inf with nan
                CETOP_value[np.isinf(CETOP_value)] = np.nan
                CETOP_values.append(CETOP_value)
            except Exception as e:
                print(f"Error while processing: {e}")
                CETOP_values.append(np.nan)
        return np.array(CETOP_values)
    def calculate_all_DASTDs_numpy(stock_data_matrix, market_returns):
        # Create weights with half-life of 42 days
        weights = 0.5 ** (np.arange(len(market_returns)) / 42)
        dastds = []
        for stock_prices in stock_data_matrix:
            # Get last 252 trading days for each stock
            last_252_days_stocks = stock_prices[-252:].astype(float)
            # Compute stock returns
            stock_returns = last_252_days_stocks[1:] / last_252_days_stocks[:-1] - 1
            # Only calculate dastd if we have enough data
            if len(stock_returns) == len(market_returns):
                # Calculate the excess returns
                excess_returns = stock_returns - market_returns
                # Calculate the weighted standard deviation of the excess returns
                weights_excess_returns_sq = weights * excess_returns ** 2 * 1e6
                dastd = np.sqrt(weights_excess_returns_sq.sum()) / 1e3
                dastds.append(dastd)
            else:
                dastds.append(np.nan)  # Append NaN for stocks with insufficient data
        return np.array(dastds)
    def calculate_all_CMRAs_numpy(stock_data_matrix):
        # Define the time period
        T = 21 * 12  # 12 months with each month having 21 trading days
        cmras = []
        for stock_prices in stock_data_matrix:
            # Get last T trading days for each stock
            last_T_days_stock = stock_prices[-T:].astype(float)
            # Compute stock returns
            stock_returns = last_T_days_stock[1:] / last_T_days_stock[:-1] - 1
            # Only calculate CMRA if we have enough data
            if len(stock_returns) == (T - 1):  # T-1 returns based on T price observations
                # Calculate cumulative returns for each month
                Z_T = [stock_returns[i: i + 21].sum() for i in range(0, len(stock_returns), 21)]
                # Check if Z_T is not empty and there's enough data to calculate CMRA
                if Z_T and len(Z_T) == 12:  # Expecting 12 monthly returns
                    # Calculate CMRA
                    cmra = np.log(1 + max(Z_T)) - np.log(1 + min(Z_T))
                    cmras.append(cmra)
                else:
                    cmras.append(np.nan)  # Append NaN for stocks with insufficient data
            else:
                cmras.append(np.nan)  # Append NaN for stocks with insufficient data
        return np.array(cmras)
    def calculate_all_HSIGMAs_numpy(stock_data_matrix, market_data, index='000300.SH', half_life=60):
        # Ensure the market data is sorted by date
        market_data_sorted = market_data[market_data[:, -1].argsort()]
        # Filter for market data
        # market_data_filtered = market_data_sorted[market_data_sorted[:, 1] == '000300.SH']
        market_data_filtered = market_data_sorted[market_data_sorted[:, 1] == index]
        # Filter the last 250 trading days for the market index
        last_250_days_market = market_data_filtered[-250:, 10].astype(float)
        # Compute market returns
        market_returns = last_250_days_market[1:] / last_250_days_market[:-1] - 1
        # Create weights with half-life of 60 days
        weights = 0.5 ** (np.arange(len(last_250_days_market) - 1) / half_life)
        hsigmas = []
        for stock_prices in stock_data_matrix:
            # Get last 250 trading days for each stock
            last_250_days_stocks = stock_prices[-250:].astype(float)
            # Compute stock returns
            stock_returns = last_250_days_stocks[1:] / last_250_days_stocks[:-1] - 1
            # Only calculate hsigma if we have enough data
            if len(stock_returns) == len(market_returns):
                # Calculate weighted covariance between stock and market
                covariance = np.cov(stock_returns, market_returns, aweights=weights)
                # Calculate variance of market
                market_var = np.var(market_returns, ddof=1)
                # Calculate the beta coefficients for the stock (covariance / market variance)
                beta = covariance[0, 1] / market_var
                # Compute residuals
                residuals = stock_returns - (beta * market_returns)
                # Compute HSIGMA: weighted standard deviation of residuals
                weighted_residuals = residuals.flatten() * weights[::-1]
                hsigma = np.sqrt(np.average((weighted_residuals - weighted_residuals.mean()) ** 2))
                hsigmas.append(hsigma)
            else:
                hsigmas.append(np.nan)  # Append NaN for stocks with insufficient data
        return np.array(hsigmas)
    def calculate_EGRO_vectorized(target_date_rank, statement_type='408001000'):
        target_date = get_trade_date_from_rank(target_date_rank)
        target_year = datetime.strptime(target_date, '%Y%m%d').year
        # Fetch AShareIncome data
        df_AShareIncome = retrieve_AShareIncome_data(statement_type)
        # Keep only the rows with 'REPORT_PERIOD' in the last 5 years
        last_five_years = df_AShareIncome[
            df_AShareIncome['REPORT_PERIOD'].astype(str).str.slice(0, 4).astype(int) >= (target_year - 5)]
        # Create an empty DataFrame with indexes of all unique stocks
        df_tot_oper_rev = pd.DataFrame(index=df_AShareIncome['S_INFO_WINDCODE'].unique(),
                                       columns=range(target_year - 5, target_year + 1))
        # For each year, find the maximum 'TOT_OPER_REV' for each stock and store in df_tot_oper_rev
        for year in range(target_year - 5, target_year + 1):
            df_year = last_five_years[last_five_years['REPORT_PERIOD'].astype(str).str.slice(0, 4).astype(int) == year]
            df_year_max = df_year.groupby('S_INFO_WINDCODE')['TOT_OPER_REV'].max()
            df_tot_oper_rev[year] = df_year_max
        # Fill NaN values with preceding values (forward fill along columns) to handle missing years
        df_tot_oper_rev = df_tot_oper_rev.ffill(axis=1)
        # Calculate the compounded growth rate for each stock
        EGRO_values = np.power(df_tot_oper_rev.iloc[:, -1] / df_tot_oper_rev.iloc[:, 0], 1 / 5) - 1
        # Convert to numpy array
        EGRO_values = EGRO_values.to_numpy()
        return EGRO_values
    def calculate_SGRO_vectorized(target_date_rank):
        # Sort the data by stock codes and date rank
        sorted_indices = np.lexsort((all_stock_data_np[:, 2], all_stock_data_np[:, 1]))
        sorted_data = all_stock_data_np[sorted_indices]
        # Get the indices where the stock code changes (which means the start of a new chunk)
        change_indices = np.where(sorted_data[:-1, 1] != sorted_data[1:, 1])[0] + 1
        # Add the start and end indices to the list
        change_indices = np.append(np.insert(change_indices, 0, 0), sorted_data.shape[0])
        # Initialize an empty list to hold the SGRO values
        SGRO_values = np.empty((len(unique_stock_codes_20230512),))
        SGRO_values[:] = np.nan
        # Create a dictionary to map stock codes to their positions in the unique_stock_codes_20230512 array
        code_to_position = {code: i for i, code in enumerate(unique_stock_codes_20230512)}
        # Iterate over the chunks
        for i in range(len(change_indices) - 1):
            # Get the start and end indices of the current chunk
            start, end = change_indices[i], change_indices[i + 1]
            # Get the data for the current stock
            stock_data = sorted_data[start:end]
            # Check if the current stock is in the unique_stock_codes_20230512 array
            stock_code = stock_data[0, 1]
            if stock_code not in code_to_position:
                SGRO_values[i] = 0
                continue
            # Get the net profits
            net_profits = stock_data[:, 26]
            # Calculate the yearly net profits
            year_net_profits = net_profits[::252]  # Select every 252th day
            # Check if there are enough data points
            if len(year_net_profits) < 3:
                SGRO_values[code_to_position[stock_code]] = 0
                continue
            # Calculate the ratio
            ratio = year_net_profits[-1] / year_net_profits[0]
            # Check for zero or negative numbers before root
            if ratio <= 0:
                SGRO_values[code_to_position[stock_code]] = 0
                continue
            # Calculate the compounded growth rate
            SGRO_values[code_to_position[stock_code]] = np.power(ratio, 1 / 3) - 1  # 3 years in this case
            non_nan_SGRO_values = SGRO_values[~np.isnan(SGRO_values)]
        return non_nan_SGRO_values
    def calculate_BTOP_vectorized(target_date_rank_12):
        # Fetch balance sheet data for all stocks
        df_AShareBalanceSheet = retrieve_AShareBalanceSheet_data('408001000')
        # Get the last year
        last_year = str(int(all_stock_data_np[all_stock_data_np[:, -1] == target_date_rank_12, 2][0][:4]) - 1)
        # Get the balance sheet data for the last year
        df_AShareBalanceSheet_selected = df_AShareBalanceSheet[
            df_AShareBalanceSheet['REPORT_PERIOD'].str.startswith(last_year)]
        # # Sort df_AShareBalanceSheet_selected based on 'S_INFO_WINDCODE' to align with all_stock_data_np
        # df_AShareBalanceSheet_selected = df_AShareBalanceSheet_selected.sort_values('S_INFO_WINDCODE')
        # Drop the rows with NaN values in 'TOT_LIAB_SHRHLDR_EQY'
        df_AShareBalanceSheet_selected = df_AShareBalanceSheet_selected.dropna(subset=['TOT_LIAB_SHRHLDR_EQY'])
        # Filter 'df_AShareBalanceSheet_selected' to keep only the row with the
        # largest 'TOT_LIAB_SHRHLDR_EQY' for each unique 'S_INFO_WINDCODE'
        df_AShareBalanceSheet_selected = df_AShareBalanceSheet_selected.loc[
            df_AShareBalanceSheet_selected.groupby('S_INFO_WINDCODE')['TOT_LIAB_SHRHLDR_EQY'].idxmax()]
        # Get 'TOT_LIAB_SHRHLDR_EQY' values for all stocks into a numpy array
        common_equity = df_AShareBalanceSheet_selected['TOT_LIAB_SHRHLDR_EQY'].to_numpy()
        # Ensure no NaN values, replace with 0
        common_equity = np.nan_to_num(common_equity, nan=0.0)
        # Get current market capitalization for all stocks into a numpy array
        current_market_cap = all_stock_data_np[:, 4]
        # Ensure no NaN or 0 values, replace with a small number to avoid division by zero
        current_market_cap = np.nan_to_num(current_market_cap, nan=1e-10)
        current_market_cap[current_market_cap == 0] = 1e-10
        # Calculate BTOP for all stocks in one operation
        n = len(common_equity)
        BTOP_values = common_equity / current_market_cap[:n]
        return BTOP_values
    def calculate_MLEV_vectorized(target_date_rank):
        # Get the specific trade date's data
        trade_date_data = all_stock_data_np[all_stock_data_np[:, -1] == target_date_rank]
        # Get current total market value of the enterprise for all stocks
        # (Assuming market cap is at index 4)
        ME = trade_date_data[:, 4]
        # Get the balance sheet data in pandas form for all stocks
        df_AShareBalanceSheet = retrieve_AShareBalanceSheet_data('408001000')  # statement_type is '408001000'
        # Get the last year
        last_year = str(int(trade_date_data[0, 2][:4]) - 1)
        # Filter the balance sheet data for the last year
        df_AShareBalanceSheet_selected = df_AShareBalanceSheet[
            df_AShareBalanceSheet['REPORT_PERIOD'] == last_year + '1231']
        # Keep only the row with the max 'TOT_LIAB' for each unique 'S_INFO_WINDCODE'
        max_index = df_AShareBalanceSheet_selected.groupby('S_INFO_WINDCODE')['TOT_LIAB'].idxmax()
        # Filter NaN values
        max_index = max_index.dropna()
        df_AShareBalanceSheet_selected = df_AShareBalanceSheet_selected.loc[max_index]
        # Sort the rows by 'S_INFO_WINDCODE' to match the order in trade_date_data
        df_AShareBalanceSheet_selected = df_AShareBalanceSheet_selected.sort_values('S_INFO_WINDCODE')
        # Get the 'TOT_LIAB' values
        LD = df_AShareBalanceSheet_selected['TOT_LIAB'].to_numpy()
        # Calculate MLEV
        n = len(LD)
        MLEV_values = (ME[:n] + LD) / ME[:n]
        return MLEV_values
    def calculate_DTOA_vectorized(target_date_rank):
        # Get the specific trade date's data
        trade_date_data = all_stock_data_np[all_stock_data_np[:, -1] == target_date_rank]
        # Get the last year
        last_year = str(int(trade_date_data[0, 2][:4]) - 1)
        # Get the balance sheet data in pandas form for all stocks
        df_AShareBalanceSheet = retrieve_AShareBalanceSheet_data('408001000')  # statement_type is '408001000'
        # Filter the balance sheet data for the last year
        df_AShareBalanceSheet_selected = df_AShareBalanceSheet[
            df_AShareBalanceSheet['REPORT_PERIOD'] == last_year + '1231']
        # Keep only the row with the max 'TOT_LIAB' and 'TOT_ASSETS' for each unique 'S_INFO_WINDCODE'
        max_index_TD = df_AShareBalanceSheet_selected.groupby('S_INFO_WINDCODE')['TOT_LIAB'].idxmax()
        max_index_TA = df_AShareBalanceSheet_selected.groupby('S_INFO_WINDCODE')['TOT_ASSETS'].idxmax()
        # Filter NaN values
        max_index_TD = max_index_TD.dropna()
        max_index_TA = max_index_TA.dropna()
        df_AShareBalanceSheet_selected_TD = df_AShareBalanceSheet_selected.loc[max_index_TD]
        df_AShareBalanceSheet_selected_TA = df_AShareBalanceSheet_selected.loc[max_index_TA]
        # Sort the rows by 'S_INFO_WINDCODE' to match the order in trade_date_data
        df_AShareBalanceSheet_selected_TD = df_AShareBalanceSheet_selected_TD.sort_values('S_INFO_WINDCODE')
        df_AShareBalanceSheet_selected_TA = df_AShareBalanceSheet_selected_TA.sort_values('S_INFO_WINDCODE')
        # Get the 'TOT_LIAB' and 'TOT_ASSETS' values
        TD = df_AShareBalanceSheet_selected_TD['TOT_LIAB'].to_numpy()
        TA = df_AShareBalanceSheet_selected_TA['TOT_ASSETS'].to_numpy()
        # Calculate DTOA
        n = len(TD)
        DTOA_values = TD / TA[:n]
        return DTOA_values
    def calculate_BLEV_vectorized(target_date_rank):
        # Get the specific trade date's data
        trade_date_data = all_stock_data_np[all_stock_data_np[:, -1] == target_date_rank]
        # Get the last year
        last_year = str(int(trade_date_data[0, 2][:4]) - 1)
        # Get the balance sheet data in pandas form for all stocks
        df_AShareBalanceSheet = retrieve_AShareBalanceSheet_data('408001000')  # statement_type is '408001000'
        # Filter the balance sheet data for the last year
        df_AShareBalanceSheet_selected = df_AShareBalanceSheet[
            df_AShareBalanceSheet['REPORT_PERIOD'] == last_year + '1231']
        # Keep only the row with the max 'TOT_LIAB' and 'TOT_ASSETS' for each unique 'S_INFO_WINDCODE'
        max_index_LD = df_AShareBalanceSheet_selected.groupby('S_INFO_WINDCODE')['TOT_LIAB'].idxmax()
        max_index_BE = df_AShareBalanceSheet_selected.groupby('S_INFO_WINDCODE')['TOT_ASSETS'].idxmax()
        # Filter NaN values
        max_index_LD = max_index_LD.dropna()
        max_index_BE = max_index_BE.dropna()
        df_AShareBalanceSheet_selected_LD = df_AShareBalanceSheet_selected.loc[max_index_LD]
        df_AShareBalanceSheet_selected_BE = df_AShareBalanceSheet_selected.loc[max_index_BE]
        # Sort the rows by 'S_INFO_WINDCODE' to match the order in trade_date_data
        df_AShareBalanceSheet_selected_LD = df_AShareBalanceSheet_selected_LD.sort_values('S_INFO_WINDCODE')
        df_AShareBalanceSheet_selected_BE = df_AShareBalanceSheet_selected_BE.sort_values('S_INFO_WINDCODE')
        # Get the 'TOT_LIAB' and 'TOT_ASSETS' values
        LD = df_AShareBalanceSheet_selected_LD['TOT_LIAB'].to_numpy()
        BE = df_AShareBalanceSheet_selected_BE['TOT_ASSETS'].to_numpy()
        # Calculate BLEV
        n = len(LD)
        BLEV_values = (BE[:n] + LD) / BE[:n]
        return BLEV_values
    def calculate_STOM_vectorized_single(target_date_rank):
        # Get the target_date from target_date_rank
        target_date = get_trade_date_from_rank(target_date_rank)
        # Fetch the turnover data for the target date
        df_AShareYield = basic_func.get_sqlserver(f"select * from AShareYield where TRADE_DT='{target_date}'", "wind")
        # Replace None values in 'TURNOVER_D' column with 0
        df_AShareYield['TURNOVER_D'] = df_AShareYield['TURNOVER_D'].fillna(1e-6)
        # Calculate STOM
        STOM_values = np.log(df_AShareYield['TURNOVER_D'].values + 1e-6)
        return STOM_values
    def calculate_STOM_vectorized(target_date_rank):
        # Initialize an array to store the results for each of the 21 days
        results = []
        max_length = 0
        for i in range(21):
            # Calculate the STOM for each of the past 21 days and append to results
            STOM_values = calculate_STOM_vectorized_single(target_date_rank - i).astype(float)
            results.append(STOM_values)
            max_length = max(max_length, len(STOM_values))
        # Initialize an array of zeros with the maximum length observed
        total_STOM_values = np.zeros(max_length, dtype=float)
        for STOM_values in results:
            # Add zeros at the end of each array to match the maximum length
            padded_values = np.pad(STOM_values, (0, max_length - len(STOM_values)), constant_values=0)
            # Add the padded array to the total
            total_STOM_values += padded_values
        return total_STOM_values
    def calculate_STOQ_vectorized(target_date_rank):
        # Set T value
        T = 3
        # Create an array of target_date_rank values
        target_date_ranks = target_date_rank - np.arange(T)
        # Initialize a container to hold the results from each period
        results = []
        max_length = 0
        for rank in target_date_ranks:
            # Calculate the STOM for the rank and append to results
            STOM_values = calculate_STOM_vectorized(rank)
            max_length = max(max_length, len(STOM_values))
            results.append(np.exp(STOM_values))
        # Pad all results to have the same length
        padded_results = []
        for result in results:
            padded_result = np.pad(result, (0, max_length - len(result)), constant_values=0)
            padded_results.append(padded_result)
        # Average the results across the T periods
        avg_results = np.mean(padded_results, axis=0)
        # Take the log of the average results to get STOQ
        STOQ_values = np.log(avg_results)
        return STOQ_values
    def calculate_STOA_vectorized(target_date_rank):
        # Set T value
        T = 12
        # Create an array of target_date_rank values
        target_date_ranks = target_date_rank - np.arange(T)
        # Initialize a container to hold the results from each period
        results = []
        max_length = 0
        for rank in target_date_ranks:
            # Calculate the STOM for the rank and append to results
            STOM_values = calculate_STOM_vectorized(rank)
            max_length = max(max_length, len(STOM_values))
            results.append(np.exp(STOM_values))
        # Pad all results to have the same length
        padded_results = []
        for result in results:
            padded_result = np.pad(result, (0, max_length - len(result)), constant_values=0)
            padded_results.append(padded_result)
        # Average the results across the T periods
        avg_results = np.mean(padded_results, axis=0)
        # Take the log of the average results to get STOQ
        STOQ_values = np.log(avg_results)
        return STOQ_values
    #################          End of risk factors         #################
    def compute_factor_matrix(split_arrays, stock_data_matrix_20230512_index_21, all_index_data_np,
                              stock_data_matrix_20230512_index_10, stock_data_matrix_20230512_index_29,
                              stock_data_matrix_20230512_index_4, target_date_rank, target_trade_date):
        # Calculate 'NLSIZE_values' first and find its length
        NLSIZE_values = np.array([calculate_NLSIZE_numpy(arr) for arr in split_arrays])
        target_length = len(NLSIZE_values)
        # Initialize the factor matrix with NLSIZE_values
        factor_matrix = NLSIZE_values.reshape(-1, 1)  # Reshape to column vector
        # List of all factors calculations
        factors_calculations = [
            calculate_all_betas_numpy(stock_data_matrix_20230512_index_21, all_index_data_np),
            np.array([calculate_RSTR_vectorized(arr) for arr in split_arrays]),
            get_last_values(calculate_LNCAP_vectorized(stock_data_matrix_20230512_index_21)),
            get_last_values(calculate_ETOP_vectorized(stock_data_matrix_20230512_index_10)),
            get_last_values(
                calculate_CETOP_vectorized(stock_data_matrix_20230512_index_29, stock_data_matrix_20230512_index_4)),
            calculate_all_DASTDs_numpy(stock_data_matrix_20230512_index_21, prepare_market_data(all_index_data_np)),
            calculate_all_CMRAs_numpy(stock_data_matrix_20230512_index_21),
            calculate_all_HSIGMAs_numpy(stock_data_matrix_20230512_index_21, all_index_data_np),
            calculate_EGRO_vectorized(target_date_rank, statement_type='408001000'),
            calculate_SGRO_vectorized(target_date_rank),
            calculate_BTOP_vectorized(target_date_rank),
            calculate_MLEV_vectorized(target_date_rank),
            calculate_DTOA_vectorized(target_date_rank),
            calculate_BLEV_vectorized(target_date_rank),
            calculate_STOM_vectorized(target_date_rank),
            calculate_STOQ_vectorized(target_date_rank),
            calculate_STOA_vectorized(target_date_rank)
        ]
        # Process each factor
        for factor_values in factors_calculations:
            # If the length of the factor values is less than the target length, pad with zeros
            if len(factor_values) < target_length:
                factor_values = np.pad(factor_values, (0, target_length - len(factor_values)), 'constant',
                                       constant_values=(0, 0))
            # If the length of the factor values is greater than the target length, truncate
            elif len(factor_values) > target_length:
                factor_values = factor_values[:target_length]
            # Append the factor values as a new column in the factor matrix
            factor_matrix = np.hstack((factor_matrix, factor_values.reshape(-1, 1)))  # Reshape to column vector
        return factor_matrix
    def normalize_factors(factor_matrix):
        # Get the column names (factors)
        factors = factor_matrix.columns
        # Normalize each factor
        for factor in factors:
            factor_values = factor_matrix[factor]
            mean = factor_values.mean()
            std = factor_values.std()
            # Replace NaN values with the mean
            factor_values.fillna(mean, inplace=True)
            # Normalize the factor values
            normalized_values = (factor_values - mean) / std
            # Update the factor values in the DataFrame
            factor_matrix[factor] = normalized_values
        return factor_matrix
    if __name__ == "__main__":
        start = time.process_time()
        all_stock_data_np_sorted = sort_data(all_stock_data_np)
        print(time.process_time() - start)
        all_stock_data_np_20230512 = all_stock_data_np[all_stock_data_np[:, 2] == target_trade_date]
        unique_stock_codes_20230512 = all_stock_data_np_20230512[:, 1]
        target_trade_date = target_trade_date
        target_date_rank = get_trade_date_rank(target_trade_date)
        col_idx_list = [21, 4, 10, 29]
        # stock_data_matrix_20230512 = create_stock_data_matrix(all_stock_data_np_sorted, target_trade_date,
        #                                              col_idx_list)
        stock_data_matrix_20230512 = np.load('stock_data_matrix_20230512.npy', allow_pickle=True)
        # print(f"stock_data_matrix_20230512 is be like: {stock_data_matrix_20230512}")
        split_arrays = split_data_by_stock_code(all_stock_data_np)
        # print(f"split_arrays is be like: {split_arrays}")
        stock_data_matrix_20230512_index_21 = retrieve_first_items(stock_data_matrix_20230512, 0)
        stock_data_matrix_20230512_index_4 = retrieve_first_items(stock_data_matrix_20230512, 1)
        stock_data_matrix_20230512_index_10 = retrieve_first_items(stock_data_matrix_20230512, 2)
        stock_data_matrix_20230512_index_29 = retrieve_first_items(stock_data_matrix_20230512, 3)
        # NLSIZE_values = np.array([calculate_NLSIZE_numpy(arr) for arr in split_arrays])
        # print(f"NLSIZE_values values are: {NLSIZE_values}, and its length is: {len(NLSIZE_values)}")
        # Beta_values = calculate_all_betas_numpy(stock_data_matrix_20230512_index_21, all_index_data_np)
        # print(f"Beta_values is: {Beta_values}, and its length is: {len(Beta_values)}")
        # RSTR_values = np.array([calculate_RSTR_vectorized(arr) for arr in split_arrays])
        # print(f"RSTR_values is: {RSTR_values}, and its length is: {len(RSTR_values)}")
        # LNCAP_values = calculate_LNCAP_vectorized(stock_data_matrix_20230512_index_21)
        # LNCAP_values = get_last_values(LNCAP_values)
        # print(f"LNCAP_values is: {LNCAP_values}, and its length is: {len(LNCAP_values)}")
        # ETOP_values = calculate_ETOP_vectorized(stock_data_matrix_20230512_index_10)
        # ETOP_values = get_last_values(ETOP_values)
        # print(f"ETOP_values is: {ETOP_values}, and its length is: {len(ETOP_values)}")
        # CETOP_values = calculate_CETOP_vectorized(stock_data_matrix_20230512_index_29, stock_data_matrix_20230512_index_4)
        # CETOP_values = get_last_values(CETOP_values)
        # print(f"CETOP_values is: {CETOP_values}, and its length is: {len(CETOP_values)}")
        # market_returns = prepare_market_data(all_index_data_np)
        # DASTD_values = calculate_all_DASTDs_numpy(stock_data_matrix_20230512_index_21, market_returns)
        # print(f"DASTD_values is: {DASTD_values}, and its length is: {len(DASTD_values)}")
        # CMRA_values = calculate_all_CMRAs_numpy(stock_data_matrix_20230512_index_21)
        # print(f"CMRA_values is: {CMRA_values}, and its length is: {len(CMRA_values)}")
        # HSIGMA_values = calculate_all_HSIGMAs_numpy(stock_data_matrix_20230512_index_21, all_index_data_np)
        # print(f"HSIGMA_values is: {HSIGMA_values}, and its length is: {len(HSIGMA_values)}")
        # target_date_rank = get_trade_date_rank(target_trade_date)
        # EGRO_values = calculate_EGRO_vectorized(target_date_rank, statement_type='408001000')
        # print(f"EGRO_values is: {EGRO_values}, and its length is: {len(EGRO_values)}")
        # SGRO_values = calculate_SGRO_vectorized(target_date_rank)
        # print(f"SGRO_values is: {SGRO_values}, and its length is: {len(SGRO_values)}")
        # BLEV_values = calculate_BLEV_vectorized(target_date_rank)
        # print(f"BLEV_values is: {BLEV_values}, and its length is: {len(BLEV_values)}")
        # STOM_values = calculate_STOM_vectorized(target_date_rank)
        # print(f"STOM_values is: {STOM_values}, and its length is: {len(STOM_values)}")
        # STOA_values = calculate_STOA_vectorized(target_date_rank)
        # print(f"STOA_values is: {STOA_values}, and its length is: {len(STOA_values)}")

        # print(f"the length of stock_data_matrix_20230512 is : {stock_data_matrix_20230512.shape}")
        # print(f" the stock_data_matrix_20230512 is be like: {stock_data_matrix_20230512}")
        # stock_data_matrix_20230512 = np.array(stock_data_matrix_20230512)
        # stock_data_matrix_2D = np.concatenate(stock_data_matrix_20230512)
        # print("the value of stock_data_matrix_20230512 is : ")
        # print(stock_data_matrix_20230512)

        # first_items = [arr[0] for arr in stock_data_matrix_20230512]
        # print(f"first_items is be like: {first_items}")

        # Usage
        compute_factor_matrix = compute_factor_matrix(split_arrays, stock_data_matrix_20230512_index_21,
                                                      all_index_data_np,
                                                      stock_data_matrix_20230512_index_10,
                                                      stock_data_matrix_20230512_index_29,
                                                      stock_data_matrix_20230512_index_4, target_date_rank,
                                                      target_trade_date)
        print(compute_factor_matrix)
        np.save(f'compute_factor_matrix_{target_trade_date}.npy', compute_factor_matrix)
        columns = ['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP',
                   'CMRA', 'DASTD', 'HSIGMA', 'EGRO', 'SGRO', 'BTOP',
                   'MLEV', 'DTOA', 'BLEV', 'STOM', 'STOQ', 'STOA']
        compute_factor_matrix = pd.DataFrame(compute_factor_matrix, columns=columns)
        compute_factor_matrix_normalized = normalize_factors(compute_factor_matrix)
        np.save(f'compute_factor_matrix_normalized_{target_trade_date}.npy', compute_factor_matrix_normalized)
        print(f"The calculation time for {target_trade_date} is {time.process_time() - start}")
    return compute_factor_matrix_normalized

def run_multiple_dates(initial_date, num_runs, rank_interval):
    initial_date_rank = get_trade_date_rank(initial_date)
    for i in range(num_runs):
        # calculate the current date rank
        target_trade_rank = initial_date_rank - i * rank_interval
        # convert rank back to date format
        target_trade_date = get_trade_date_from_rank(target_trade_rank)
        print(f"target_trade_rank on {target_trade_date} is {target_trade_rank}")
        run_the_main(target_trade_date)
def get_multiple_dates(initial_date, num_runs, rank_interval):
    initial_date_rank = get_trade_date_rank(initial_date)
    multiple_target_dates = []
    for i in range(num_runs):
        # calculate the current date rank
        target_trade_rank = initial_date_rank - i * rank_interval
        # convert rank back to date format
        target_trade_date = get_trade_date_from_rank(target_trade_rank)
        print(f"target_trade_rank on {target_trade_date} is {target_trade_rank}")
        multiple_target_dates.append(target_trade_date)
    return multiple_target_dates
def check_files_exist(multiple_target_dates):
    missing_files = [] # A list to store the dates for which the file does not exist
    for date in multiple_target_dates:
        filename = f'compute_factor_matrix_normalized_{date}.npy'
        if not os.path.isfile(filename):
            missing_files.append(date)
    return missing_files
# if __name__ == "__main__":
#     initial_trade_date = '20230512'
#     num_runs = 252
#     rank_interval = 1
    # run_multiple_dates(initial_trade_date, num_runs, rank_interval)

if __name__ == "__main__":
    initial_trade_date = '20230512'
    num_runs = 365
    rank_interval = 1
    multiple_target_dates = get_multiple_dates(initial_trade_date, num_runs, rank_interval)

    missing_files = check_files_exist(multiple_target_dates)
    np.save('missing_files.npy', missing_files)
    if missing_files:
        print("Files not found for the following dates:")
        for date in missing_files:
            print(date)
    else:
        print("All files exist.")

