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



# Use the function to retrieve data
# beginDate = '2018.05.09'
beginDate = '2020.01.01'
endDate = '2023.05.12'
all_stock_data, all_index_data = retrieve_data(beginDate, endDate)




# # Save the retrieved data to pickle files for future use
# all_stock_data.to_pickle("./all_stock_data.pkl")
# all_index_data.to_pickle("./all_index_data.pkl")

# # Load the pickle file into a DataFrame
# all_stock_data = pd.read_pickle("./all_stock_data.pkl") # AShareEODDerivativeIndicator
# all_index_data = pd.read_pickle("./all_index_data.pkl") # AIndexEODPrices
# # Save the DataFrame as a text file
# all_stock_data.to_csv("./all_stock_data.txt", sep='\t')
# all_index_data.to_csv("./all_index_data.txt", sep='\t')


if __name__ == "__main__":

    # Convert pandas DataFrame to numpy array
    all_stock_data_np = all_stock_data.values
    all_stock_data_np = all_stock_data_np[all_stock_data_np[:, 1].argsort()]

    all_index_data_np = all_index_data.values
    all_index_data_np = all_index_data_np[all_index_data_np[:, 1].argsort()]

    # with open('./all_stock_data_np.pkl', 'rb') as f:
    #     all_stock_data_np = pickle.load(f)
    #
    # with open('./all_index_data_np.pkl', 'rb') as f:
    #     all_index_data_np = pickle.load(f)

    # print(all_stock_data_np)

    np.save('all_stock_data_np.npy', all_stock_data_np)
    np.save('all_index_data_np.npy', all_index_data_np)

    # all_stock_data_np = np.load('all_stock_data_np.npy')
    # all_index_data_np = np.load('all_index_data_np.npy')

    # print(all_stock_data_np)





