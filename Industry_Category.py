from my.data import basic_func
import pandas as pd
import numpy as np
import pickle
from datetime import datetime



# General formula settings
def list_stocks(target_date, all_stock_data):
    # Convert DataFrame columns to numpy arrays
    dates = all_stock_data['TRADE_DT'].to_numpy()
    stock_codes = all_stock_data['S_INFO_WINDCODE'].to_numpy()
    # Get boolean mask of the given date
    mask = dates == target_date
    # Filter stock codes with the mask and get unique ones
    unique_stock_codes = np.unique(stock_codes[mask])
    return unique_stock_codes
def list_stock_industry(all_stock_data_IND):
    stock_codes = all_stock_data_IND['S_INFO_WINDCODE'].unique()
    # Initialize df_stock_industry_category
    df_stock_industry_category = pd.DataFrame()
    for stock_code in stock_codes:
        df_temp = basic_func.get_sqlserver(f"select * from AShareSECNIndustriesClass where S_INFO_WINDCODE = '{stock_code}'", "wind")
        df_temp = df_temp[df_temp['CUR_SIGN'] == 1]
        # Append df_temp to df_stock_industry_category
        df_stock_industry_category = df_stock_industry_category.append(df_temp, ignore_index=True)
    # Get the first 4 digits of 'SEC_IND_CODE'
    df_stock_industry_category['SEC_IND_CODE_SHORT'] = df_stock_industry_category['SEC_IND_CODE'].astype(str).str[:4]
    # # Map SEC_IND_CODE_SHORT values
    # all_stock_data_IND['SEC_IND_CODE_SHORT'] = all_stock_data_IND['S_INFO_WINDCODE'].map(df_stock_industry_category.set_index('S_INFO_WINDCODE')['SEC_IND_CODE_SHORT'])
    # # Only fill NaN values in SEC_IND_CODE_SHORT with '1000'
    # all_stock_data_IND['SEC_IND_CODE_SHORT'].fillna('1219', inplace=True)
    # all_stock_data_IND = all_stock_data_IND.drop_duplicates(subset='S_INFO_WINDCODE', keep='last')
    return df_stock_industry_category

if __name__ == "__main__":
#######################################     This is for retrieving industry data     #########################################
    # Load the data
    # all_stock_data = pd.read_pickle('./all_stock_data.pkl')
    with open('./all_stock_data_np.pkl', 'rb') as f:
        all_stock_data = pickle.load(f)
    print('Data successfully loaded.')
    all_stock_data_updated = list_stock_industry(all_stock_data)
    print(all_stock_data_updated)
    print(len(all_stock_data_updated))
    # Save the retrieved data to pickle files for future use
    all_stock_data_updated.to_pickle("./all_stock_industry_data_df.pkl")
    # Save it as txt
    all_stock_data_updated.to_csv("./all_stock_industry_data_df.txt", sep='\t')


#######################################     This is for classifying industry data     #########################################

    # Load the pickled DataFrame
    df = pd.read_pickle("./all_stock_industry_data_df.pkl")
    # Convert the DataFrame to a numpy array
    all_stock_industry_data_np = df.values
    np.save('all_stock_industry_data_np.npy', all_stock_industry_data_np)
    # print(all_stock_industry_data_np)
    # Get the last column which represents 'SEC_IND_CODE_SHORT'
    industry_codes = all_stock_industry_data_np[:, [1,-1]]
    np.save('all_stock_unique_industry_codes.npy', industry_codes)
    short_industry_codes = all_stock_industry_data_np[:, -1]
    # Get all unique industry codes
    unique_industry_codes = np.unique(short_industry_codes)

    # Print all unique industry codes
    # print(industry_codes[industry_codes[:, 0] == '000001.SZ'])
    print(industry_codes)
    print(unique_industry_codes)
