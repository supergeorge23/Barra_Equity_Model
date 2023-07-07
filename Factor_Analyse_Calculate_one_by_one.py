import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime
from my.data import basic_func
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "5"



def factor_analysis_main():
    import pandas as pd
    import numpy as np
    import os
    from sklearn.linear_model import LinearRegression
    from datetime import datetime
    from my.data import basic_func
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    all_stock_data_np_full = np.load('all_stock_data_np.npy', allow_pickle=True)
    # Create a boolean mask for the condition that the stock code does not end with '.BJ'
    mask = np.array([not code.endswith('.BJ') for code in all_stock_data_np_full[:, 1]])

    # Apply the mask to the array
    all_stock_data_np = all_stock_data_np_full[mask]

    all_index_data_np = np.load('all_index_data_np.npy', allow_pickle=True)
    print("Data successfully loaded.")

    # General formula settings
    def calculate_market_return(index_codes_0):
        # Filter the data for the selected indexes
        filtered_data = all_index_data_np[np.isin(all_index_data_np[:, 1], index_codes_0)]
        # Calculate the mean of 'S_DQ_PCTCHANGE' grouped by 'TRADE_DT'
        # Here, I'm assuming 'TRADE_DT' is sorted in ascending order. If not, you'd need to sort the data first
        dates, index = np.unique(filtered_data[:, 2], return_inverse=True)
        index = index.astype(int)
        weights = filtered_data[:, 10].astype(float)
        return np.bincount(index, weights=weights) / np.bincount(index)

    def get_trade_date_rank(target_date_0):
        # Retrieve the 'trade_date_rank' for the target_date
        target_date_rank_0 = all_stock_data_np[all_stock_data_np[:, 2] == target_date_0, -1][0]
        return target_date_rank_0

    def get_trade_date_from_rank(target_date_rank_0):
        # Convert the target_date_rank to an actual trading date
        target_date_0 = all_stock_data_np[all_stock_data_np[:, -1] == target_date_rank_0, 2][0]
        return target_date_0

    ### Factor 1: NLSIZE
    def calculate_single_NLSIZE(stock_code_1, target_date_rank_1):
        try:
            # Get the market value data for the specific stock
            stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_1]
            # Ensure the market value data is numeric and does not contain zero or negative values
            market_value = stock_data[:, 4].astype(np.float)  # Convert to float
            market_value[market_value <= 0] = np.finfo(
                np.float64).tiny  # Replace zero or negative values with the smallest positive number
            # Calculate the logarithmic market value
            log_market_value = np.log(market_value)
            # Remove or replace NaN and inf values
            log_market_value = log_market_value[~np.isinf(log_market_value)]  # Removes -inf, inf
            log_market_value = log_market_value[~np.isnan(log_market_value)]  # Removes NaN
            # Create the regressor which is the cube of the logarithmic market value
            cubic_log_market_value = log_market_value ** 3
            # Create the model
            model = LinearRegression()
            # Reshape the data to fit the model
            cubic_log_market_value_reshaped = cubic_log_market_value.reshape(-1, 1)
            log_market_value_reshaped = log_market_value.reshape(-1, 1)
            # Fit the model with the logarithmic market value as weights
            model.fit(cubic_log_market_value_reshaped, log_market_value_reshaped, sample_weight=log_market_value)
            # Calculate the residuals
            residuals_sum = (log_market_value - model.predict(cubic_log_market_value_reshaped).ravel()).sum()
            return residuals_sum
        except Exception as e:
            # print(f"Error calculating NLSIZE for stock {stock_code_1}: {e}")
            return 0

    ### Factor 2: Beta (Length of Yield = 250 , Half-life = 60)
    def calculate_single_beta(stock_code_2, target_date_rank_2):
        # Filter for the specific stock
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_2]
        # Ensure the data is sorted by date
        stock_data = stock_data[stock_data[:, -1].argsort()]
        # Calculate the stock return
        stock_returns = stock_data[1:, 21] / stock_data[:-1, 21] - 1
        # Filter the last 250 trading days for the stock
        last_250_days_stock = stock_returns[stock_data[:-1, -1] <= target_date_rank_2][-250:]
        # Filter the data for the market index '000300.SH'
        market_data = all_index_data_np[all_index_data_np[:, 1] == '000300.SH']
        # Ensure the data is sorted by date
        market_data = market_data[market_data[:, -1].argsort()]
        # Filter the last 250 trading days for the market index
        last_250_days_market = market_data[market_data[:, -1] <= target_date_rank_2][-250:, 10]
        # # Check if we have enough data
        if len(last_250_days_market) < 250 or len(last_250_days_stock) < 250:
            return 0, 0, 0
        # Create weights with half-life of 60 days
        weights = 0.5 ** (np.arange(len(last_250_days_stock)) / 60)
        # Create an array of stock returns
        X = last_250_days_stock.reshape(-1, 1)
        # Create an array of market returns
        y = last_250_days_market.reshape(-1, 1)
        # Create linear regression object
        reg = LinearRegression()
        # Fit the linear regression using the weights
        reg.fit(X, y, sample_weight=weights)
        beta = reg.coef_[0][0]
        alpha = reg.intercept_[0]
        # Calculate residuals
        residuals = y - reg.predict(X)
        return beta, alpha, residuals

    ### Factor 3: RSTR (T = 500 , L = 21)
    def calculate_single_RSTR(stock_code_3, target_date_rank_3):
        L = 21
        T = 500
        # Get the specific stock's data
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_3]
        # Ensure the data is sorted by date (trade_date_rank is assumed to be the last column)
        stock_data = stock_data[stock_data[:, -1].argsort()]
        # Select the subset of data for T+L days before the target date
        period_stock = stock_data[stock_data[:, -1] <= target_date_rank_3][-T - L:]
        # Calculate the return, which is the rate of change in closing price
        returns = np.log((period_stock[1:, 21].astype(float) / period_stock[:-1, 21].astype(float)))
        # Generate weights with half-life of 120 days
        weights = np.power(0.5 ** (1 / 120), np.arange(T + L - 1, -1, -1))
        # Select the last T entries of the returns and weights
        selected_returns = returns[-T:].sum()
        selected_weights = weights[:120]
        # Calculate RSTR
        RSTR_3 = (selected_returns * selected_weights).sum()
        return RSTR_3

    ### Factor 4: LNCAP
    def calculate_single_LNCAP(stock_code_4, target_date_rank_4):
        # Filter the stock data for the specified stock and date
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_4]
        stock_data_on_target_date = stock_data[stock_data[:, -1] == target_date_rank_4]
        # If there is no data for the stock on the target date, return None
        if stock_data_on_target_date.size == 0:
            return 0
        # Calculate LNCAP
        LNCAP_4 = np.log(stock_data_on_target_date[0, 4])  # Assuming 'S_VAL_MV' is at index 4
        return LNCAP_4

    ### Factor 5: ETOP
    def calculate_single_ETOP(stock_code_5, target_date_rank_5):
        # Filter the stock data for the specified stock and date
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_5]
        stock_data_on_target_date = stock_data[stock_data[:, -1] == target_date_rank_5]
        # If there is no data for the stock on the target date, return 0
        if stock_data_on_target_date.size == 0:
            return 0
        # Calculate ETOP
        ETOP_5 = 1 / stock_data_on_target_date[0, 10]  # Assuming 'S_VAL_PE_TTM' is at index 10
        return ETOP_5

    ### Factor 6: CETOP
    def calculate_single_CETOP(stock_code_6, target_date_rank_6):
        # Filter the stock data for the specified stock and date
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_6]
        stock_data_on_target_date = stock_data[stock_data[:, -1] == target_date_rank_6]
        # If there is no data for the stock on the target date, return 0
        if stock_data_on_target_date.size == 0:
            return 0
        # Calculate CETOP
        net_cash_flow = stock_data_on_target_date[0, 29]  # Assuming 'NET_CASH_FLOWS_OPER_ACT_TTM' is at index 30
        market_cap = stock_data_on_target_date[0, 4]  # Assuming 'S_VAL_MV' is at index 4
        CETOP_6 = net_cash_flow / market_cap
        return CETOP_6

    ### Factor 7: DASTD
    def calculate_single_DASTD(stock_code_7, target_date_rank_7):
        T_7 = 252
        half_life_7 = 42
        # Get the data for the specific stock and calculate the return
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_7]
        stock_returns = np.diff(stock_data[:, 5]) / stock_data[:-1, 5]  # Assuming 'S_DQ_CLOSE_TODAY' is at index 5
        # Get the data for the market index and calculate the return
        index_data = all_index_data_np[all_index_data_np[:, 1] == '000300.SH']
        index_returns = index_data[:, 5]  # Assuming 'S_DQ_PCTCHANGE' is at index 5
        # Make sure the two arrays have the same size
        min_length = min(len(stock_returns), len(index_returns))
        stock_returns = stock_returns[-min_length:]
        index_returns = index_returns[-min_length:]
        # Calculate the excess returns
        excess_returns = stock_returns - index_returns
        # Create weights with a half-life of 42 days
        weights = 0.5 ** (np.arange(len(excess_returns)) / half_life_7)
        # Calculate the weighted standard deviation of the excess returns
        weights_excess_returns_sq_7 = weights * excess_returns ** 2 * 1e6
        DASTD = np.sqrt(weights_excess_returns_sq_7.sum()) / 1e3
        return DASTD

    ### Factor 8: CMRA
    def calculate_single_CMRA(stock_code_8, target_date_rank_8):
        # Define the time period
        T_8 = 21 * 12  # 12 months with each month having 21 trading days
        # Filter the stock data for the specified stock and date
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_8]
        try:
            # Calculate the return series for the stock over the specified time period
            stock_returns = (stock_data[target_date_rank_8 - T_8 + 1: target_date_rank_8, 5] /
                             stock_data[target_date_rank_8 - T_8: target_date_rank_8 - 1, 5] - 1)
            # Calculate cumulative returns for each month
            Z_T = [stock_returns[i: i + 21].sum() for i in range(0, len(stock_returns), 21)]
            # Check if Z_T is empty
            if len(Z_T) > 0:
                # Calculate CMRA
                CMRA_8 = np.log(1 + max(Z_T)) - np.log(1 + min(Z_T))
            else:
                CMRA_8 = 0  # or a predefined value indicating insufficient data
        except Exception as e:
            # print(f"Error in calculating 'stock_returns': {e}")
            CMRA_8 = 0  # Return 0 in case of an error
        return CMRA_8

    ### Factor 9: HSIGMA
    def calculate_single_HSIGMA(stock_code_9, target_date_rank_9):
        # Get beta, alpha, and residuals from the beta calculation function
        beta, alpha, residuals = calculate_single_beta(stock_code_9, target_date_rank_9)
        # Define the time period and half-life
        T_9 = 252  # Number of trading days
        half_life_9 = 63  # Half-life of weights
        # Create an array of weights with a half-life decay
        weights = np.array([0.5 ** (1 / half_life_9)]).cumprod() ** np.arange(T_9)
        weights = weights[:, np.newaxis]
        # Check the type of residuals before attempting to flatten
        if isinstance(residuals, np.ndarray):
            weighted_residuals = residuals.flatten() * weights[::-1]
        else:
            # Handle the case where residuals is not an array-like
            # print(f"Residuals for stock {stock_code_9} is not an array. It's a {type(residuals)}")
            return 0
        # Calculate the weighted standard deviation of the residuals
        hsigma_9 = np.sqrt(np.average(((weighted_residuals - weighted_residuals.mean()) ** 2).sum()))
        return hsigma_9

    ### Factor 10: EGRO
    def calculate_single_EGRO(stock_code_10, target_date_rank_9):
        target_date_10 = get_trade_date_from_rank(target_date_rank_9)
        # Get the current year
        target_year_10 = datetime.strptime(target_date_10, '%Y%m%d').year
        # Get the data for the specific stock
        df_AShareIncome = basic_func.get_sqlserver(
            f"select * from AShareIncome where S_INFO_WINDCODE='{stock_code_10}'", "wind")
        # Keep only the rows with 'REPORT_PERIOD' in the last 5 years
        last_five_years = df_AShareIncome[df_AShareIncome['REPORT_PERIOD'] >= str((int(target_year_10) - 5)) + '0101']
        # Initialize a list to store the total operating income for each year
        year_tot_oper_rev = []
        # Calculate the yearly total operating income
        for year in range(int(target_year_10) - 5, int(target_year_10)):
            # Filter the data for the specific year
            year_data = last_five_years[last_five_years['REPORT_PERIOD'].str.startswith(str(year))]
            # If there is data for the year, find the maximum total operating income and append it to the list
            if len(year_data) > 0:
                max_tot_oper_rev = year_data['TOT_OPER_REV'].max()
                year_tot_oper_rev.append(max_tot_oper_rev)
            # If there is no data for the year, append a NaN to the list
            else:
                year_tot_oper_rev.append(np.nan)
        # Convert the list to a numpy array
        year_tot_oper_rev = np.array(year_tot_oper_rev)
        # Calculate the compounded growth rate
        EGRO_10 = np.power(year_tot_oper_rev[-1] / year_tot_oper_rev[0], 1 / 5) - 1
        return EGRO_10

    ### Factor 11: SGRO
    def calculate_single_SGRO(stock_code_11, target_date_rank_11):
        # Get the data for the specific stock
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_11]
        # Suppose 'NET_PROFIT_PARENT_COMP_TTM' is at index 22
        net_profits = stock_data[target_date_rank_11 - 252 * 3: target_date_rank_11, 26]
        # Check if there are enough data points
        if len(net_profits) < 252 * 3:
            # print(f"Not enough data to calculate SGRO for stock {stock_code_11}")
            return 0
        # Calculate the yearly net profits
        year_net_profits = [net_profits[i: i + 252][-1] for i in range(0, len(net_profits), 252)]
        # Check for zero before division
        if year_net_profits[0] == 0:
            # print(f"Cannot calculate SGRO for stock {stock_code_11} due to division by zero")
            return 0
        # Calculate the ratio
        ratio = year_net_profits[-1] / year_net_profits[0]
        # Check for negative numbers before root
        if ratio < 0:
            # print(f"Cannot calculate SGRO for stock {stock_code_11} due to negative ratio")
            return 0
        # Calculate the compounded growth rate
        SGRO = np.power(ratio, 1 / 5) - 1
        return SGRO

    ### Factor 12: BTOP
    def calculate_single_BTOP(stock_code_12, target_date_rank_12):
        # Get the specific stock's data
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_12]
        # Ensure the data is sorted by date (trade_date_rank is assumed to be the last column)
        stock_data = stock_data[stock_data[:, -1].argsort()]
        # Get current market capitalization
        # (Assuming market cap is at index 4)
        current_market_cap = stock_data[stock_data[:, -1] == target_date_rank_12, 4][0]
        # Get the balance sheet data in pandas form
        df_AShareBalanceSheet = basic_func.get_sqlserver(
            f"select * from AShareBalanceSheet where S_INFO_WINDCODE='{stock_code_12}'", "wind")
        # Get the common equity for the past year
        last_year = str(int(stock_data[stock_data[:, -1] == target_date_rank_12, 2][0][:4]) - 1)
        common_equity_df = df_AShareBalanceSheet[df_AShareBalanceSheet['REPORT_PERIOD'].str.startswith(last_year)][
            'TOT_LIAB_SHRHLDR_EQY']
        # Convert pandas Series to numpy array
        common_equity = common_equity_df.to_numpy()
        # Check if any values were obtained for common_equity
        if common_equity.size == 0:
            # print(f"No common equity data available for stock {stock_code_12} for the year {last_year}")
            return 0
        # Get the max common equity
        common_equity_max = common_equity.max()
        # Calculate BTOP
        BTOP = common_equity_max / current_market_cap
        return BTOP

    ### Factor 13: MLEV
    def calculate_single_MLEV(stock_code_13, target_date_rank_13):
        # Get the specific stock's data
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_13]
        # Ensure the data is sorted by date (trade_date_rank is assumed to be the last column)
        stock_data = stock_data[stock_data[:, -1].argsort()]
        # Get current total market value of the enterprise
        # (Assuming market cap is at index 22)
        ME = stock_data[stock_data[:, -1] == target_date_rank_13, 22][0]
        # Get the balance sheet data in pandas form
        df_AShareBalanceSheet = basic_func.get_sqlserver(
            f"select * from AShareBalanceSheet where S_INFO_WINDCODE='{stock_code_13}'", "wind")
        # Get the long-term debt for the past year
        last_year = str(int(stock_data[stock_data[:, -1] == target_date_rank_13, 2][0][:4]) - 1)
        LD_df = df_AShareBalanceSheet[df_AShareBalanceSheet['REPORT_PERIOD'].str.startswith(last_year)]['TOT_LIAB']
        # Convert pandas Series to numpy array
        LD = LD_df.to_numpy()
        # Check if any values were obtained for LD
        if LD.size == 0:
            print(f"No long-term debt data available for stock {stock_code_13} for the year {last_year}")
            return None
        # Get the max long-term debt
        LD_max = LD.max()
        # Calculate MLEV
        MLEV = (ME + LD_max) / ME
        return MLEV

    ### Factor 14: DTOA
    def calculate_single_DTOA(stock_code_14, target_date_rank_14):
        # Get the specific stock's data
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_14]
        # Ensure the data is sorted by date (trade_date_rank is assumed to be the last column)
        stock_data = stock_data[stock_data[:, -1].argsort()]
        # Get the balance sheet data in pandas form
        df_AShareBalanceSheet = basic_func.get_sqlserver(
            f"select * from AShareBalanceSheet where S_INFO_WINDCODE='{stock_code_14}'", "wind")
        # Get the total liabilities and total assets for the past year
        last_year = str(int(stock_data[stock_data[:, -1] == target_date_rank_14, 2][0][:4]) - 1)
        TD_df = df_AShareBalanceSheet[df_AShareBalanceSheet['REPORT_PERIOD'].str.startswith(last_year)]['TOT_LIAB']
        TA_df = df_AShareBalanceSheet[df_AShareBalanceSheet['REPORT_PERIOD'].str.startswith(last_year)]['TOT_ASSETS']
        # Convert pandas Series to numpy arrays
        TD = TD_df.to_numpy()
        TA = TA_df.to_numpy()
        # Check if any values were obtained for TD and TA
        if TD.size == 0 or TA.size == 0:
            # print(f"No total liabilities or total assets data available for stock {stock_code_14} for the year {last_year}")
            return 0
        # Get the max total liabilities and total assets
        TD_max = TD.max()
        TA_max = TA.max()
        # Calculate DTOA
        DTOA = TD_max / TA_max
        return DTOA

    ### Factor 15: BLEV
    def calculate_single_BLEV(stock_code_15, target_date_rank_15):
        # Get the data for the specific stock
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code_15]
        # Ensure the data is sorted by date (trade_date_rank is assumed to be the last column)
        stock_data = stock_data[stock_data[:, -1].argsort()]
        # Get the balance sheet data
        df_AShareBalanceSheet = basic_func.get_sqlserver(
            f"select * from AShareBalanceSheet where S_INFO_WINDCODE='{stock_code_15}'", "wind")
        # Get the total liabilities and total assets for the past year
        last_year = int(int(stock_data[stock_data[:, -1] == target_date_rank_15][:, 2][0]) / 10000) - 1

        LD = df_AShareBalanceSheet[df_AShareBalanceSheet['REPORT_PERIOD'].str.startswith(str(last_year))][
            'TOT_LIAB'].max()
        BE = df_AShareBalanceSheet[df_AShareBalanceSheet['REPORT_PERIOD'].str.startswith(str(last_year))][
            'TOT_ASSETS'].max()
        # Convert LD and BE into numpy arrays
        LD = np.array(LD)
        BE = np.array(BE)
        # Calculate BLEV
        BLEV = (BE + LD) / BE
        return BLEV

    ### Factor 16: STOM
    def calculate_single_STOM(stock_code_16, target_date_rank_16):
        # Get the yield data
        df_AShareYield = basic_func.get_sqlserver(f"select * from AShareYield where S_INFO_WINDCODE='{stock_code_16}'",
                                                  "wind")
        # Convert the 'TRADE_DT' to datetime
        df_AShareYield['TRADE_DT'] = pd.to_datetime(df_AShareYield['TRADE_DT'], format='%Y%m%d')
        # Convert the target_date_rank to an actual trading date
        target_date = get_trade_date_from_rank(target_date_rank_16)
        target_date = pd.to_datetime(target_date, format='%Y%m%d')
        # Select only the data in the 21 days period leading up to (and including) the target date
        df_AShareYield = df_AShareYield[(df_AShareYield['TRADE_DT'] <= target_date) &
                                        (df_AShareYield['TRADE_DT'] > target_date - pd.Timedelta(days=21))]
        # Calculate the daily turnover for the past 21 days
        TO_t = df_AShareYield['TURNOVER_D'].values
        # Convert None to 0 in TO_t
        TO_t = [0 if v is None else v for v in TO_t]
        # If there is no turnover data for the past 21 days, return 0
        if len(TO_t) == 0 or np.sum(TO_t) == 0:
            return 0
        # Calculate STOM
        STOM = np.log(np.sum(TO_t))
        return STOM

    ### Factor 17: STOQ
    def calculate_single_STOQ(stock_code_17, target_date_rank_17):
        # Set T value
        T_17 = 3
        # Create an array of target_date_rank values
        target_date_ranks = target_date_rank_17 - np.arange(T_17)
        # Use a list comprehension to apply calculate_single_STOM to each date rank
        STOM_values = [calculate_single_STOM(stock_code_17, rank) for rank in target_date_ranks]
        # Calculate and return STOQ
        STOQ = np.log(np.mean(np.exp(STOM_values)))
        return STOQ

    ### Factor 18: STOA
    def calculate_single_STOA(stock_code_18, target_date_rank_18):
        # Set T value
        T_18 = 12
        # Create an array of target_date_rank values
        target_date_ranks = target_date_rank_18 - np.arange(T_18)
        # Use a list comprehension to apply calculate_single_STOM to each date rank
        STOM_values = [calculate_single_STOM(stock_code_18, rank) for rank in target_date_ranks]
        # Calculate and return STOQ
        STOQ = np.log(np.mean(np.exp(STOM_values)))
        return STOQ

    ### Calculate the factor return
    def list_stocks(target_date):
        # Filter the data for the given date
        data_on_target_date = all_stock_data_np[all_stock_data_np[:, 2] == target_date]
        # Get all unique stock codes
        unique_stock_codes_str = np.unique(data_on_target_date[:, 1])
        print("length of unique_stock_codes_str is: ")
        print(len(unique_stock_codes_str))
        print("unique_stock_codes_str is: ")
        print(unique_stock_codes_str)
        # Split the concatenated string into separate stock codes
        # unique_stock_codes = np.array([unique_stock_codes_str[i:i+9] for i in range(0, len(unique_stock_codes_str), 9)])
        # unique_stock_codes = [unique_stock_codes_str[i:i+1] for i in range(0, len(unique_stock_codes_str), 9)]
        unique_stock_codes = [unique_stock_codes_str[i] for i in range(len(unique_stock_codes_str))]
        np.save('20230512_unique_stock_codes.npy', unique_stock_codes)
        return unique_stock_codes

    # ## Calculate in safe
    #
    # # Define a new function that wraps around your original function and handles exceptions
    # def calculate_NLSIZE_for_all_stocks(stock_codes, calculate_single_NLSIZE):
    #     # Prepare an empty numpy array to hold the NLSIZE values
    #     NLSIZE_values = np.empty_like(stock_codes, dtype=float)
    #     # Loop through each stock code
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             # Try to calculate the NLSIZE
    #             NLSIZE_values[i] = calculate_single_NLSIZE(stock_code)
    #         except Exception as e:
    #             # If an error occurs, print the error and assign 0 to the NLSIZE value
    #             print(f"Error calculating NLSIZE for {stock_code}: {e}")
    #             NLSIZE_values[i] = 0
    #     return NLSIZE_values
    #
    #
    # def calculate_beta_for_all_stocks(stock_codes, calculate_single_beta, target_date_rank_2):
    #     # Prepare empty numpy arrays to hold the beta, alpha and residuals
    #     beta_values = np.empty_like(stock_codes, dtype=float)
    #     # alpha_values = np.empty_like(stock_codes, dtype=float)
    #     # residuals_values = np.empty((len(stock_codes), 250), dtype=float)
    #     # Loop through each stock code
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             # Try to calculate the beta, alpha and residuals
    #             beta_values[i], _, _ = calculate_single_beta(stock_code, target_date_rank_2)
    #             # beta_values[i], alpha_values[i], residuals = calculate_single_beta(stock_code, target_date_rank_2)
    #             # residuals_values[i] = residuals.flatten()
    #         except Exception as e:
    #             print(f"Error calculating beta for {stock_code}: {e}")
    #             beta_values[i] = 0
    #             # alpha_values[i] = 0
    #             # residuals_values[i] = np.zeros(250)
    #     return beta_values
    #
    #
    # def calculate_RSTR_for_all_stocks(stock_codes, calculate_single_RSTR, target_date_rank):
    #     RSTR_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             RSTR_values[i] = calculate_single_RSTR(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating RSTR for {stock_code}: {e}")
    #             RSTR_values[i] = 0
    #     return RSTR_values
    #
    #
    # def calculate_LNCAP_for_all_stocks(stock_codes, calculate_single_LNCAP, target_date_rank):
    #     LNCAP_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             LNCAP_values[i] = calculate_single_LNCAP(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating LNCAP for {stock_code}: {e}")
    #             LNCAP_values[i] = 0
    #     return LNCAP_values
    #
    #
    # def calculate_ETOP_for_all_stocks(stock_codes, calculate_single_ETOP, target_date_rank):
    #     ETOP_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             ETOP_values[i] = calculate_single_ETOP(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating ETOP for {stock_code}: {e}")
    #             ETOP_values[i] = 0
    #     return ETOP_values
    #
    #
    # def calculate_CETOP_for_all_stocks(stock_codes, calculate_single_CETOP, target_date_rank):
    #     CETOP_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             CETOP_values[i] = calculate_single_CETOP(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating CETOP for {stock_code}: {e}")
    #             CETOP_values[i] = 0
    #     return CETOP_values
    #
    #
    # def calculate_DASTD_for_all_stocks(stock_codes, calculate_single_DASTD):
    #     DASTD_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             DASTD_values[i] = calculate_single_DASTD(stock_code)
    #         except Exception as e:
    #             print(f"Error calculating DASTD for {stock_code}: {e}")
    #             DASTD_values[i] = 0
    #     return DASTD_values
    #
    #
    # def calculate_CMRA_for_all_stocks(stock_codes, calculate_single_CMRA, target_date_rank):
    #     CMRA_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             CMRA_values[i] = calculate_single_CMRA(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating CMRA for {stock_code}: {e}")
    #             CMRA_values[i] = 0
    #     return CMRA_values
    #
    #
    # def calculate_HSIGMA_for_all_stocks(stock_codes, calculate_single_HSIGMA, target_date_rank):
    #     HSIGMA_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             HSIGMA_values[i] = calculate_single_HSIGMA(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating HSIGMA for {stock_code}: {e}")
    #             HSIGMA_values[i] = 0
    #     return HSIGMA_values
    #
    #
    # def calculate_EGRO_for_all_stocks(stock_codes, calculate_single_EGRO, target_date):
    #     EGRO_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             EGRO_values[i] = calculate_single_EGRO(stock_code, target_date)
    #         except Exception as e:
    #             print(f"Error calculating EGRO for {stock_code}: {e}")
    #             EGRO_values[i] = 0
    #     return EGRO_values
    #
    #
    # def calculate_SGRO_for_all_stocks(stock_codes, calculate_single_SGRO, target_date_rank):
    #     SGRO_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             SGRO_values[i] = calculate_single_SGRO(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating SGRO for {stock_code}: {e}")
    #             SGRO_values[i] = 0
    #     return SGRO_values
    #
    #
    # def calculate_BTOP_for_all_stocks(stock_codes, calculate_single_BTOP, target_date_rank):
    #     BTOP_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             BTOP_values[i] = calculate_single_BTOP(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating BTOP for {stock_code}: {e}")
    #             BTOP_values[i] = 0
    #     return BTOP_values
    #
    #
    # def calculate_MLEV_for_all_stocks(stock_codes, calculate_single_MLEV, target_date_rank):
    #     MLEV_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             MLEV_values[i] = calculate_single_MLEV(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating MLEV for {stock_code}: {e}")
    #             MLEV_values[i] = 0
    #     return MLEV_values
    #
    #
    # def calculate_DTOA_for_all_stocks(stock_codes, calculate_single_DTOA, target_date_rank):
    #     DTOA_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             DTOA_values[i] = calculate_single_DTOA(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating DTOA for {stock_code}: {e}")
    #             DTOA_values[i] = 0
    #     return DTOA_values
    #
    #
    # def calculate_BLEV_for_all_stocks(stock_codes, calculate_single_BLEV, target_date_rank):
    #     BLEV_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             BLEV_values[i] = calculate_single_BLEV(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating BLEV for {stock_code}: {e}")
    #             BLEV_values[i] = 0
    #     return BLEV_values
    #
    #
    # def calculate_STOM_for_all_stocks(stock_codes, calculate_single_STOM, target_date_rank):
    #     STOM_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             STOM_values[i] = calculate_single_STOM(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating STOM for {stock_code}: {e}")
    #             STOM_values[i] = 0
    #     return STOM_values
    #
    #
    # def calculate_STOQ_for_all_stocks(stock_codes, calculate_single_STOQ, target_date_rank):
    #     STOQ_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             STOQ_values[i] = calculate_single_STOQ(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating STOQ for {stock_code}: {e}")
    #             STOQ_values[i] = 0
    #     return STOQ_values
    #
    #
    # def calculate_STOA_for_all_stocks(stock_codes, calculate_single_STOA, target_date_rank):
    #     STOA_values = np.empty_like(stock_codes, dtype=float)
    #     for i, stock_code in enumerate(stock_codes):
    #         try:
    #             STOA_values[i] = calculate_single_STOA(stock_code, target_date_rank)
    #         except Exception as e:
    #             print(f"Error calculating STOA for {stock_code}: {e}")
    #             STOA_values[i] = 0
    #     return STOA_values
    #
    #
    # def calculate_all_factors_parallel(stock_codes,
    #                                    calculate_single_NLSIZE,
    #                                    calculate_single_beta,
    #                                    calculate_single_RSTR,
    #                                    calculate_single_LNCAP,
    #                                    calculate_single_ETOP,
    #                                    calculate_single_CETOP,
    #                                    calculate_single_DASTD,
    #                                    calculate_single_CMRA,
    #                                    calculate_single_HSIGMA,
    #                                    calculate_single_EGRO,
    #                                    calculate_single_SGRO,
    #                                    calculate_single_BTOP,
    #                                    calculate_single_MLEV,
    #                                    calculate_single_DTOA,
    #                                    calculate_single_BLEV,
    #                                    calculate_single_STOM,
    #                                    calculate_single_STOQ,
    #                                    calculate_single_STOA,
    #                                    target_date_rank,
    #                                    target_date):
    #     with ProcessPoolExecutor(max_workers=18) as executor:
    #         future_NLSIZE = executor.submit(calculate_NLSIZE_for_all_stocks, stock_codes, calculate_single_NLSIZE)
    #         future_beta = executor.submit(calculate_beta_for_all_stocks, stock_codes, calculate_single_beta,
    #                                       target_date_rank)
    #         future_RSTR = executor.submit(calculate_RSTR_for_all_stocks, stock_codes, calculate_single_RSTR,
    #                                       target_date_rank)
    #         future_LNCAP = executor.submit(calculate_LNCAP_for_all_stocks, stock_codes, calculate_single_LNCAP,
    #                                        target_date_rank)
    #         future_ETOP = executor.submit(calculate_ETOP_for_all_stocks, stock_codes, calculate_single_ETOP,
    #                                       target_date_rank)
    #         future_CETOP = executor.submit(calculate_CETOP_for_all_stocks, stock_codes, calculate_single_CETOP,
    #                                        target_date_rank)
    #         future_DASTD = executor.submit(calculate_DASTD_for_all_stocks, stock_codes, calculate_single_DASTD)
    #         future_CMRA = executor.submit(calculate_CMRA_for_all_stocks, stock_codes, calculate_single_CMRA,
    #                                       target_date_rank)
    #         future_HSIGMA = executor.submit(calculate_HSIGMA_for_all_stocks, stock_codes, calculate_single_HSIGMA,
    #                                         target_date_rank)
    #         future_EGRO = executor.submit(calculate_EGRO_for_all_stocks, stock_codes, calculate_single_EGRO,
    #                                       target_date)
    #         future_SGRO = executor.submit(calculate_SGRO_for_all_stocks, stock_codes, calculate_single_SGRO,
    #                                       target_date_rank)
    #         future_BTOP = executor.submit(calculate_BTOP_for_all_stocks, stock_codes, calculate_single_BTOP,
    #                                       target_date_rank)
    #         future_MLEV = executor.submit(calculate_MLEV_for_all_stocks, stock_codes, calculate_single_MLEV,
    #                                       target_date_rank)
    #         future_DTOA = executor.submit(calculate_DTOA_for_all_stocks, stock_codes, calculate_single_DTOA,
    #                                       target_date_rank)
    #         future_BLEV = executor.submit(calculate_BLEV_for_all_stocks, stock_codes, calculate_single_BLEV,
    #                                       target_date_rank)
    #         future_STOM = executor.submit(calculate_STOM_for_all_stocks, stock_codes, calculate_single_STOM,
    #                                       target_date_rank)
    #         future_STOQ = executor.submit(calculate_STOQ_for_all_stocks, stock_codes, calculate_single_STOQ,
    #                                       target_date_rank)
    #         future_STOA = executor.submit(calculate_STOA_for_all_stocks, stock_codes, calculate_single_STOA,
    #                                       target_date_rank)
    #
    #     print(f"future_NLSIZE value length is: {future_NLSIZE}")
    #     NLSIZE_results = future_NLSIZE.result()
    #     print(f"future_beta value length is: {future_beta}")
    #     beta_results = future_beta.result()
    #     print(f"future_RSTR value length is: {future_RSTR}")
    #     RSTR_results = future_RSTR.result()
    #     print(f"future_LNCAP value length is: {future_LNCAP}")
    #     LNCAP_results = future_LNCAP.result()
    #     print(f"future_ETOP value length is: {future_ETOP}")
    #     ETOP_results = future_ETOP.result()
    #     print(f"future_CETOP value length is: {future_CETOP}")
    #     CETOP_results = future_CETOP.result()
    #     print(f"future_DASTD value length is: {future_DASTD}")
    #     DASTD_results = future_DASTD.result()
    #     print(f"future_CMRA value length is: {future_CMRA}")
    #     CMRA_results = future_CMRA.result()
    #     print(f"future_HSIGMA value length is: {future_HSIGMA}")
    #     HSIGMA_results = future_HSIGMA.result()
    #     print(f"future_EGRO value length is: {future_EGRO}")
    #     EGRO_results = future_EGRO.result()
    #     print(f"future_EGRO value length is: {future_SGRO}")
    #     SGRO_results = future_SGRO.result()
    #     print(f"future_BTOP value length is: {future_BTOP}")
    #     BTOP_results = future_BTOP.result()
    #     print(f"future_MLEV value length is: {future_MLEV}")
    #     MLEV_results = future_MLEV.result()
    #     print(f"future_DTOA value length is: {future_DTOA}")
    #     DTOA_results = future_DTOA.result()
    #     print(f"future_BLEV value length is: {future_BLEV}")
    #     BLEV_results = future_BLEV.result()
    #     print(f"future_STOM value length is: {future_STOM}")
    #     STOM_results = future_STOM.result()
    #     print(f"future_STOQ value length is: {future_STOQ}")
    #     STOQ_results = future_STOQ.result()
    #     print(f"future_STOA value length is: {future_STOA}")
    #     STOA_results = future_STOA.result()
    #
    #     # Ensure the results are in the form of numpy arrays
    #     NLSIZE_results = np.array(NLSIZE_results)
    #     beta_results = np.array(beta_results)
    #     RSTR_results = np.array(RSTR_results)
    #     LNCAP_results = np.array(LNCAP_results)
    #     ETOP_results = np.array(ETOP_results)
    #     CETOP_results = np.array(CETOP_results)
    #     DASTD_results = np.array(DASTD_results)
    #     CMRA_results = np.array(CMRA_results)
    #     HSIGMA_results = np.array(HSIGMA_results)
    #     EGRO_results = np.array(EGRO_results)
    #     SGRO_results = np.array(SGRO_results)
    #     BTOP_results = np.array(BTOP_results)
    #     MLEV_results = np.array(MLEV_results)
    #     DTOA_results = np.array(DTOA_results)
    #     BLEV_results = np.array(BLEV_results)
    #     STOM_results = np.array(STOM_results)
    #     STOQ_results = np.array(STOQ_results)
    #     STOA_results = np.array(STOA_results)
    #
    #     # Combine the stock codes, NLSIZE results and beta results into one numpy array
    #     results_array = np.column_stack((stock_codes, NLSIZE_results, beta_results, RSTR_results, LNCAP_results,
    #                                      ETOP_results, CETOP_results, DASTD_results, CMRA_results, HSIGMA_results,
    #                                      EGRO_results, SGRO_results, BTOP_results, MLEV_results, DTOA_results,
    #                                      BLEV_results, STOM_results, STOQ_results, STOA_results))
    #     np.save('20230512_results_array.npy', results_array)
    #     return results_array
    #

    def calculate_factors_for_stock(stock_code, target_date_rank):
        factors = ['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP', 'DASTD', 'CMRA', 'HSIGMA', 'EGRO', 'SGRO',
                   'BTOP', 'MLEV', 'DTOA', 'BLEV', 'STOM', 'STOQ', 'STOA']
        factor_functions = [calculate_single_NLSIZE, calculate_single_beta, calculate_single_RSTR,
                            calculate_single_LNCAP, calculate_single_ETOP, calculate_single_CETOP,
                            calculate_single_DASTD, calculate_single_CMRA, calculate_single_HSIGMA,
                            calculate_single_EGRO, calculate_single_SGRO, calculate_single_BTOP, calculate_single_MLEV,
                            calculate_single_DTOA, calculate_single_BLEV, calculate_single_STOM, calculate_single_STOQ,
                            calculate_single_STOA]
        results = {'stock_code': stock_code}
        for factor, func in zip(factors, factor_functions):
            try:
                result = func(stock_code, target_date_rank)
                # For 'beta', we only store the first value
                if factor == 'Beta':
                    results[factor] = result[0]
                else:
                    results[factor] = result
            except Exception as e:
                print(f"Error calculating {factor} for {stock_code}: {e}")
                results[factor] = 0
        return results

    def calculate_all_factors_slow(stock_codes, target_date_rank):
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_stock_code = {
                executor.submit(calculate_factors_for_stock, stock_code, target_date_rank): stock_code for stock_code in
                stock_codes}
            for future in as_completed(future_to_stock_code):
                stock_code = future_to_stock_code[future]
                try:
                    result = future.result()
                    print(f"Factors for {stock_code}: {result}")
                    results.append(result)  # append result to results list
                except Exception as exc:
                    print(f"Error calculating factors for {stock_code}: {exc}")
        return results

    def calculate_all_factors_old(stock_codes, target_date_rank):
        all_results = []
        for stock_code in stock_codes:
            try:
                result = calculate_factors_for_stock(stock_code, target_date_rank)
                print(f"Factors for {stock_code}: {result}")
                all_results.append(result)
            except Exception as exc:
                print(f"Error calculating factors for {stock_code}: {exc}")
        return all_results

    def calculate_all_factors(stock_codes, target_date_rank):
        def process_stock(stock_code):
            try:
                result = calculate_factors_for_stock(stock_code, target_date_rank)
                print(f"Factors for {stock_code}: {result}")
                return result
            except Exception as exc:
                print(f"Error calculating factors for {stock_code}: {exc}")
                return None  # or some other default value

        # Use map to apply process_stock to each stock code
        all_results = list(map(process_stock, stock_codes))
        return all_results

    def normalize_factors(results_df):
        # Extract the keys for the risk factors
        risk_factors = [name for name in results_df.columns if name != 'stock_code']
        # Normalize each factor
        for factor in risk_factors:
            factor_values = results_df[factor]
            # Compute the mean and standard deviation
            mean = factor_values.mean()
            std = factor_values.std()
            # Normalize the factor values
            normalized_values = (factor_values - mean) / std
            # Update the factor values in the DataFrame
            results_df[factor] = normalized_values
            # Replace NaN values with 0
            results_df[factor].fillna(0, inplace=True)
        return results_df

    if __name__ == "__main__":
        # Call the calculate_all_factor_returns function and get the result
        target_date = '20230411'
        # stock_code = '000001.SZ'
        index_code = '000300.SH'
        target_date_rank = get_trade_date_rank(target_date)



        # Formal
        unique_stock_codes = list_stocks(target_date)
        print(f"Unique stock codes are: {unique_stock_codes}")

        # # Experimental
        # unique_stock_codes = np.load('20230512_unique_stock_codes.npy', allow_pickle=True)
        # print(f"Unique stock codes are: {unique_stock_codes}")

        # Testing
        # results_array = calculate_all_factors(unique_stock_codes, target_date_rank)

        all_stock_data_np_20230512 = all_stock_data_np[all_stock_data_np[:, 2] == '20230411']
        unique_stock_codes_20230512 = all_stock_data_np_20230512[:, 1]
        unique_stock_codes_20230512_short = unique_stock_codes_20230512[0:10]

        target_date_rank = all_stock_data_np_20230512[:, -1]
        target_date_rank_short = target_date_rank[0:10]

        start = time.process_time()

        # Create a vectorized version of your function
        calculate_factors_for_stock_vectorized = np.vectorize(calculate_factors_for_stock,
                                                              excluded=['target_date_rank'])

        # Call the vectorized function on the array of stock codes
        results_array = calculate_factors_for_stock_vectorized(unique_stock_codes_20230512, target_date_rank=1197)




        print(time.process_time() - start)
        start_2 = time.process_time()

        np.save('20230411_results_array.npy', results_array)
        print("results_array is: ")
        print(results_array)

        print(time.process_time() - start_2)
        start_3 = time.process_time()



        ### Nomalize factor returns
        # Convert your list of dictionaries to a DataFrame
        results_df = pd.DataFrame.from_records(results_array)

        # Use the function
        normalized_results_df = normalize_factors(results_df)

        np.save('20230411_normalized_results_array.npy', normalized_results_df)
        print("normalized_results_df is: ")
        print(normalized_results_df)

        print(time.process_time() - start_3)


if __name__ == "__main__":
    start = time.process_time()
    factor_analysis_main()
    print(time.process_time() - start)



    # # Experimental
    # unique_stock_codes = np.load('20230512_unique_stock_codes.npy', allow_pickle=True)
    # print(f"Unique stock codes are: {unique_stock_codes}")
    # print(unique_stock_codes[100])
