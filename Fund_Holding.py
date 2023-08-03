import numpy as np
import pandas as pd
import cvxpy as cp
from my.data import basic_func
from Factor_Calculator import get_multiple_dates, run_the_main
import matplotlib.pyplot as plt
all_stock_data_np = np.load('all_stock_data_np.npy', allow_pickle=True)
unique_stock_codes = np.unique(all_stock_data_np[:, 1])
print("data succcessfully loaded!")


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
def get_pct_changes(date_str, unique_stock_codes):
    df_stock = basic_func.get_sqlserver(f"select * from AShareEODPrices where trade_dt='{date_str}'", "wind")
    stock_return = df_stock[['S_DQ_PCTCHANGE', 'S_INFO_WINDCODE']]
    stock_pct_changes = get_stock_pct_change(unique_stock_codes, stock_return)
    return stock_pct_changes

def call_Fund_Holding(date_str, last_target_weights):
    from Factor_Calculator import get_multiple_dates, run_the_main
    def save_AIndexHS300_stock_codes(date_str, filename):
        df_stock = basic_func.get_sqlserver(f"select * from AIndexHS300CloseWeight where TRADE_DT='{date_str}'", "wind")
        AIndexHS300_Constituent_Stock_Code = df_stock['S_CON_WINDCODE']
        np.save(filename, AIndexHS300_Constituent_Stock_Code)
        print(AIndexHS300_Constituent_Stock_Code)
        return AIndexHS300_Constituent_Stock_Code
    def get_ranks_of_stock_codes(unique_stock_codes, stock_codes_to_find):
        # This will give a boolean array of the same shape as `unique_stock_codes`,
        # where True indicates that the corresponding item in `unique_stock_codes`
        # is in `stock_codes_to_find`.
        isIn = np.isin(unique_stock_codes, stock_codes_to_find)
        # Now we find the indices where `isIn` is True.
        ranks = np.where(isIn)[0]
        return ranks
    def get_rows_by_ranks(matrix, ranks):
        return matrix[ranks, :]
    def compute_column_means(matrix):
        return np.nanmean(matrix, axis=0)
    def calculate_stock_returns(stock_codes, all_stock_data_np, trade_date_rank):
        stock_returns = []
        for stock_code in stock_codes:
            # Calculate and store the return rates
            stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock_code]
            current_day_close_price = stock_data[stock_data[:, -1] == trade_date_rank, 21].astype(float)
            previous_day_close_price = stock_data[stock_data[:, -1] == trade_date_rank - 1, 21].astype(float)
            if current_day_close_price.size > 0 and previous_day_close_price.size > 0:
                stock_return = current_day_close_price[0] / previous_day_close_price[0] - 1
                stock_returns.append(stock_return)
            else:
                stock_returns.append(0)  # fallback if return rate can't be calculated
        return np.array(stock_returns)
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
    def get_pct_changes(date_str, unique_stock_codes):
        df_stock = basic_func.get_sqlserver(f"select * from AShareEODPrices where trade_dt='{date_str}'", "wind")
        stock_return = df_stock[['S_DQ_PCTCHANGE', 'S_INFO_WINDCODE']]
        stock_pct_changes = get_stock_pct_change(unique_stock_codes, stock_return)
        return stock_pct_changes
    def get_benchmark_weights(AIndexHS300_2_unique_stock_codes_ranks, total_stocks=5264):
        n_bench_stocks = len(AIndexHS300_2_unique_stock_codes_ranks)
        bench_weights = np.zeros(total_stocks)
        bench_weights[AIndexHS300_2_unique_stock_codes_ranks] = 1 / n_bench_stocks
        return bench_weights
    def update_stock_industry_codes(unique_stock_codes, all_stock_industry_data_np):
        # Create a dictionary from all_stock_industry_data_np
        industry_data_dict = {row[0]: row[1] for row in all_stock_industry_data_np}
        # Initialize a new array to hold the updated industry codes
        updated_stock_industry_data = []
        # Iterate over unique_stock_codes
        for stock_code in unique_stock_codes:
            # Check if the stock code is in the industry data dictionary
            if stock_code in industry_data_dict:
                # If it is, append the stock code and its industry code to the new array
                updated_stock_industry_data.append([stock_code, industry_data_dict[stock_code]])
            else:
                # If it isn't, append the stock code and the default industry code '1220' to the new array
                updated_stock_industry_data.append([stock_code, '1220'])
        # Convert the updated_stock_industry_data to a numpy array
        updated_stock_industry_data_np = np.array(updated_stock_industry_data)
        return updated_stock_industry_data_np
    def calculate_factor_exposure(weights, factor_matrix):
        # Initialize an empty list to store the average exposures
        average_exposure = []
        # Loop over each factor in the factor matrix
        for i in range(factor_matrix.shape[1]):
            # Select the factor
            factor = factor_matrix[:, i]
            # Calculate the total exposure to the factor
            total_exposure = weights.T @ factor
            # Calculate the average exposure to the factor
            average_exposure.append(total_exposure / 5264)
        # return np.array(average_exposure)
        return total_exposure
    def calculate_industry_matrix(all_stock_industry_data_np):
        # Get all unique industry codes
        unique_industry_codes = np.unique(all_stock_industry_data_np[:, 1])
        # Initialize an empty matrix
        industry_matrix = np.zeros((len(all_stock_industry_data_np), len(unique_industry_codes)))
        # Loop over all stocks
        for i, stock in enumerate(all_stock_industry_data_np):
            # Find the index of the stock's industry in the list of unique industry codes
            industry_index = np.where(unique_industry_codes == stock[1])[0][0]
            # Set the corresponding entry in the industry matrix to 1
            industry_matrix[i, industry_index] = 1
        return industry_matrix
    def portfolio_optimization_old(stock_returns, factor_matrix, bench_weights, max_weight=1, min_weight=0):
        n = len(stock_returns)
        # Define the weights variable
        weights = cp.Variable(n)
        # Define the objective function
        objective = cp.Maximize(weights.T @ stock_returns)
        bench_average_exposure = calculate_factor_exposure(bench_weights, factor_matrix)
        industry_matrix = calculate_industry_matrix(all_stock_industry_data_np)
        bench_industry_exposure = bench_weights @ industry_matrix
        portfolio_industry_exposure = weights.T @ industry_matrix
        # Define the constraints
        constraints = [
            sum(weights) <= 1,  # The weights must sum to 1
            sum(weights) > 0,  # The weights must larger than 1
            weights >= min_weight,  # The weights must be non-negative
            weights <= max_weight,  # The weights cannot exceed the max_weight
            calculate_factor_exposure(weights, factor_matrix) == bench_average_exposure,
            # Factor exposures must match benchmark
            portfolio_industry_exposure == bench_industry_exposure,  # Industry exposure must match benchmark
            weights <= 0.03  # The weight of an individual stock cannot exceed 3%
        ]
        # Define the problem
        problem = cp.Problem(objective, constraints)
        # Solve the problem
        problem.solve()
        # Return the optimal weights
        return weights.value

    def portfolio_optimization(stock_returns, factor_matrix, bench_weights, last_target_weights, max_weight=1,
                               min_weight=0, max_change=0.1, reg_param=0.01):
        n = len(stock_returns)
        # Define the weights variable
        weights = cp.Variable(n) # Initialize weights with last_target_weights
        # Define the objective function
        objective = cp.Maximize(weights.T @ stock_returns - reg_param * cp.norm(weights - last_target_weights, 2))
        bench_average_exposure = calculate_factor_exposure(bench_weights, factor_matrix)
        industry_matrix = calculate_industry_matrix(all_stock_industry_data_np)
        bench_industry_exposure = bench_weights @ industry_matrix
        portfolio_industry_exposure = weights.T @ industry_matrix
        # Define the constraints
        constraints = [
            sum(weights) <= 1,  # The weights must sum to 1
            sum(weights) > 0,  # The weights must non-negative
            weights >= min_weight,  # The weights must be non-negative
            weights <= max_weight,  # The weights cannot exceed the max_weight
            calculate_factor_exposure(weights, factor_matrix) == bench_average_exposure,
            # Factor exposures must match benchmark
            portfolio_industry_exposure == bench_industry_exposure,  # Industry exposure must match benchmark
            weights <= 0.03,  # The weight of an individual stock cannot exceed 3%
            sum(cp.abs(weights - last_target_weights)) <= max_change
            # The sum of absolute changes in weights cannot exceed 0.1
        ]
        # Define the problem
        problem = cp.Problem(objective, constraints)
        # Solve the problem
        problem.solve()
        # Return the optimal weights
        return weights.value

    if __name__ == "__main__":
        ################################################################################################################
        unique_stock_codes = np.unique(all_stock_data_np[:, 1])
        date_str = date_str
        pct_changes_20230512 = get_pct_changes(date_str, unique_stock_codes)
        # ### This is for individual date
        # compute_factor_matrix_normalized_20230512 = run_the_main(date_str)
        ### This is for old
        compute_factor_matrix_normalized_20230512 = np.load(f'compute_factor_matrix_normalized_{date_str}.npy',
                                                            allow_pickle=True)
        # This is for error
        # compute_factor_matrix_normalized_20230512 = np.load(f'coefficient_df_{date_str}.npy',
        #                                                     allow_pickle=True)
        all_stock_industry_data_np = np.load('all_stock_industry_data_np.npy', allow_pickle=True)
        all_stock_industry_data_np = update_stock_industry_codes(unique_stock_codes, all_stock_industry_data_np)
        ################################################################################################################
        AIndexHS300_Constituent_Stock_Code_20230512 = np.load(f'AIndexHS300_Constituent_Stock_Code_20230512.npy',
                                                              allow_pickle=True)
        AIndexHS300_2_unique_stock_codes_ranks = get_ranks_of_stock_codes(unique_stock_codes,
                                                                          AIndexHS300_Constituent_Stock_Code_20230512)
        bench_weights_ranks = get_benchmark_weights(AIndexHS300_2_unique_stock_codes_ranks, total_stocks=5264)
        ################################################################################################################
        portfolio_optimization_outcome = portfolio_optimization(pct_changes_20230512,
                                                                compute_factor_matrix_normalized_20230512,
                                                                bench_weights_ranks,
                                                                last_target_weights,
                                                                max_weight=1, min_weight=0,
                                                                max_change=0.1,  reg_param=0.01)
        # portfolio_optimization_outcome = np.abs(portfolio_optimization_outcome)
        portfolio_optimization_outcome = portfolio_optimization_outcome
        np.save(f'portfolio_optimization_outcome_{date_str}.npy', portfolio_optimization_outcome)
        print(f"portfolio_optimization_outcome is be like: \n {portfolio_optimization_outcome}"
              f"\n and its length is {len(portfolio_optimization_outcome)}")
        print(f"maximum of portfolio_optimization_outcome is: {portfolio_optimization_outcome.max()} "
              f"\n and maximum's rank is {np.argmax(portfolio_optimization_outcome)}"
              f"\n minimum of portfolio_optimization_outcome is: {portfolio_optimization_outcome.min()}"
              f"\n and minimum's rank is {np.argmin(portfolio_optimization_outcome)}")
        optimized_rate_of_return = pct_changes_20230512.T @ portfolio_optimization_outcome
        print(f"optimized_rate_of_return on date {date_str} is {optimized_rate_of_return}")
    return optimized_rate_of_return, portfolio_optimization_outcome, unique_stock_codes, get_pct_changes

if __name__ == "__main__":
    # Load initial weights
    last_target_weights = np.load('begin_weight.npy', allow_pickle=True)
    # last_target_weights = last_target_weights.T[0]
    # Generate multiple target dates
    initial_date = '20230512'
    ### This is for monthly data
    # multiple_target_date_list = get_multiple_dates(initial_date, num_runs=23, rank_interval=22)
    multiple_target_date_list = get_multiple_dates(initial_date, num_runs=364, rank_interval=1)
    print(f"multiple_target_date_list is be like: \n {multiple_target_date_list}")
    # Prepare lists for storing results
    multiple_date_call_Fund_Holding = []
    multiple_date_portfolio_optimization_outcome = []
    for stock_code in reversed(multiple_target_date_list):  # iterate over dates in the list
        print(f"current stock_code is: {stock_code}")
        try:
            # Call the portfolio_optimization function
            single_date_call_Fund_Holding, \
            single_date_portfolio_optimization_outcome, \
            unique_stock_codes, \
            get_pct_changes \
                = call_Fund_Holding(stock_code, last_target_weights)
            print(f"current single_date_call_Fund_Holding value is: {single_date_call_Fund_Holding}")
            print(f"single_date_portfolio_optimization_outcome is: \n {single_date_portfolio_optimization_outcome}")
            # Update last_target_weights for the next iteration
            last_target_weights = single_date_portfolio_optimization_outcome
        except Exception as e:
            print(f"Error during optimization for date {stock_code}: {e}")
            print("Using last target date's portfolio weight to calculate this date's rate of return.")
            # Calculation of rate of return with last_target_weights in case of an error
            single_date_call_Fund_Holding = np.sum(
                last_target_weights * get_pct_changes(stock_code, unique_stock_codes))
            # Keep the same weights
            single_date_portfolio_optimization_outcome = last_target_weights
        # Store results
        multiple_date_call_Fund_Holding.append(single_date_call_Fund_Holding)
        multiple_date_portfolio_optimization_outcome.append(single_date_portfolio_optimization_outcome)

    # print(f"single_date_call_Fund_Holding is be like: \n {multiple_date_call_Fund_Holding}")
    # np.save('multiple_date_call_Fund_Holding_2.npy', multiple_date_call_Fund_Holding)
    multiple_date_call_Fund_Holding_values = [x.item() if isinstance(x, np.matrix) else x for x in
                                              multiple_date_call_Fund_Holding]
    multiple_date_call_Fund_Holding_values[np.isnan(multiple_date_call_Fund_Holding_values)] = 0
    print(f"single_date_call_Fund_Holding is be like: \n {multiple_date_call_Fund_Holding_values}")
    try:
        np.save('multiple_date_call_Fund_Holding_2.npy', multiple_date_call_Fund_Holding_values)
        np.save('multiple_date_portfolio_optimization_outcome_2.npy', multiple_date_portfolio_optimization_outcome)
        print("File saved successfully.")
    except Exception as e:
        print(f"An error occurred when saving the file: {e}")

    # Convert your multiple_date_portfolio_optimization_outcome to a numpy array for easier indexing
    multiple_date_portfolio_optimization_outcome_np_old = np.array(multiple_date_call_Fund_Holding_values)
    # Reverse the order of rows
    multiple_date_portfolio_optimization_outcome_np = multiple_date_portfolio_optimization_outcome_np_old[::-1]
    multiple_target_date_list = multiple_target_date_list[::-1]
    # Create a new figure
    plt.figure()
    # Generate x-values (the dates)
    x_values = range(len(multiple_target_date_list))
    # Generate y-values (the optimization outcomes)
    # For example, we could plot the sum of the weights at each date:
    y_values = [np.sum(weights) for weights in multiple_date_portfolio_optimization_outcome_np]
    # Plot the line chart
    plt.plot(x_values, y_values)
    # Add a title
    plt.title('Portfolio optimization outcome over time')
    # Add x and y labels
    plt.xlabel('Date')
    plt.ylabel('Portfolio Return')
    # Optionally, you can set the xticks to be the actual dates for better readability
    # plt.xticks(x_values, multiple_target_date_list, rotation=45)
    plt.xticks(x_values, multiple_target_date_list, rotation=45)
    # Show the plot
    plt.tight_layout()
    plt.show()
    # Save the plot
    plt.savefig('portfolio_optimization_outcome.png')


