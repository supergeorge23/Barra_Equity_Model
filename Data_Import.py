from my.data import basic_func
import pandas as pd
import numpy as np
import pickle

def retrieve_data_old(start_date, end_date):
    # Convert strings to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Generate all dates within the time dimension
    dates = pd.date_range(start=start_date, end=end_date)
    # Containers to hold all stock and index data
    all_stock_data = []
    all_index_data = []
    # Iterate over all dates
    for date in dates:
        # Convert date to string in 'yyyymmdd' format for SQL query
        date_str = date.strftime('%Y%m%d')
        # Query stock data
        df_stock = basic_func.get_sqlserver(f"select * from AShareEODDerivativeIndicator where trade_dt='{date_str}'",
                                            "wind")
        # Query index data
        df_index = basic_func.get_sqlserver(f"select * from AIndexEODPrices where trade_dt='{date_str}'", "wind")
        # Append to the containers
        all_stock_data.append(df_stock)
        all_index_data.append(df_index)
    # Concatenate all dataframes
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    all_index_data = pd.concat(all_index_data, ignore_index=True)

    # Sort the data by trading days
    all_stock_data.sort_values('TRADE_DT', inplace=True)
    all_index_data.sort_values('TRADE_DT', inplace=True)

    # Add 'trade_date_rank' column to both dataframes
    all_stock_data = all_stock_data.assign(trade_date_rank=np.arange(1, len(all_stock_data) + 1))
    all_index_data = all_index_data.assign(trade_date_rank=np.arange(1, len(all_index_data) + 1))

    return all_stock_data, all_index_data

def retrieve_data(start_date, end_date):
    # Convert strings to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Generate all dates within the time dimension
    dates = pd.date_range(start=start_date, end=end_date)
    # Containers to hold all stock and index data
    all_stock_data = []
    all_index_data = []
    # Create dictionary to store trade date ranks
    trade_date_rank_dict = {date_str: i+1 for i, date_str in enumerate(map(lambda x: x.strftime('%Y%m%d'), dates))}
    # Iterate over all dates
    for date in dates:
        # Convert date to string in 'yyyymmdd' format for SQL query
        date_str = date.strftime('%Y%m%d')
        # Query stock data
        df_stock = basic_func.get_sqlserver(f"select * from AShareEODDerivativeIndicator where trade_dt='{date_str}'",
                                            "wind")
        # Query index data
        df_index = basic_func.get_sqlserver(f"select * from AIndexEODPrices where trade_dt='{date_str}'", "wind")
        # Add trade_date_rank column
        if not df_stock.empty:
            df_stock['trade_date_rank'] = trade_date_rank_dict[date_str]
            all_stock_data.append(df_stock)
        if not df_index.empty:
            df_index['trade_date_rank'] = trade_date_rank_dict[date_str]
            all_index_data.append(df_index)
        if not df_stock.empty:
            df_stock['trade_date_rank'] = trade_date_rank_dict[date_str]
            all_stock_data.append(df_stock)
        if not df_index.empty:
            df_index['trade_date_rank'] = trade_date_rank_dict[date_str]
            all_index_data.append(df_index)
    # Concatenate all dataframes
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    all_index_data = pd.concat(all_index_data, ignore_index=True)
    # Sort the data by trading days
    all_stock_data.sort_values('TRADE_DT', inplace=True)
    all_index_data.sort_values('TRADE_DT', inplace=True)
    return all_stock_data, all_index_data

def create_stock_data_matrix(sorted_data, price_col_idx=21):
    # Identify unique stock codes
    unique_stock_codes = np.unique(sorted_data[:, 1])
    print("unique_stock_codes has been collected!")
    # Initialize an empty list to hold the data for each stock
    stock_data_list = []
    # For each unique stock code, extract its price data and append to the list
    for stock_code in unique_stock_codes:
        stock_data = sorted_data[sorted_data[:, 1] == stock_code][:, price_col_idx]
        stock_data_list.append(stock_data)
    # Convert the list to a numpy array and transpose it
    # So that each column represents a unique stock's price data
    stock_data_matrix = np.array(stock_data_list).T
    np.save('stock_data_matrix_20230512.npy', stock_data_matrix)
    print("create_stock_data_matrix has been finished")
    return stock_data_matrix

# Use the function to retrieve data
# beginDate = '2018.05.09'
beginDate = '2020.01.01'
endDate = '2023.05.12'
all_stock_data, all_index_data = retrieve_data(beginDate, endDate)


if __name__ == "__main__":
    # Convert pandas DataFrame to numpy array
    all_stock_data_np = all_stock_data.values
    all_stock_data_np = all_stock_data_np[all_stock_data_np[:, 1].argsort()]

    all_index_data_np = all_index_data.values
    all_index_data_np = all_index_data_np[all_index_data_np[:, 1].argsort()]

    np.save('all_stock_data_np.npy', all_stock_data_np)
    np.save('all_index_data_np.npy', all_index_data_np)






