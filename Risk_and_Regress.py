import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.api as sm


# Load the data
normalized_results_array = np.load('20230512_normalized_results_array.npy', allow_pickle=True)
all_stock_data_np = np.load('all_stock_data_np.npy', allow_pickle=True)
industry_codes = np.load('all_stock_unique_industry_codes.npy', allow_pickle=True)



def parameter_estimation_OLS(trade_date_rank):
    # Extracting risk factors
    risk_factor_names = ['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP', 'DASTD', 'CMRA', 'HSIGMA',
                         'EGRO', 'SGRO', 'BTOP', 'MLEV', 'DTOA', 'BLEV', 'STOM', 'STOQ', 'STOA']
    factor_loadings = []
    stock_returns = []
    for stock in normalized_results_array:
        # Extract factor loadings
        factors = [stock[risk_factor] for risk_factor in risk_factor_names]
        # Construct industry dummy variables
        industry_code_for_stock = industry_codes[industry_codes[:, 0] == stock['stock_code'], 1][0]
        industry_dummy_variables = [1 if industry_code == industry_code_for_stock else 0 for industry_code in
                                    map(str, range(1201, 1220))]
        # Add factor loadings and dummy variables to matrix
        factor_loadings.append(factors + industry_dummy_variables)
        # Calculate and store the return rates
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock['stock_code']]
        current_day_close_price = stock_data[stock_data[:, -1] == trade_date_rank, 21].astype(float)
        previous_day_close_price = stock_data[stock_data[:, -1] == trade_date_rank - 1, 21].astype(float)
        stock_return = current_day_close_price / previous_day_close_price - 1
        stock_returns.append(stock_return[0])
    factor_loadings = np.array(factor_loadings)
    stock_returns = np.array(stock_returns)
    # Now you can use the factor_loadings as X and stock_returns as y in your linear regression model
    model = LinearRegression().fit(factor_loadings, stock_returns)
    # Define all factor names
    all_factor_names = risk_factor_names + ['Ind_' + str(i) for i in range(1201, 1220)]
    # Create a dictionary that pairs each coefficient with its corresponding factor name
    coefficient_dict = dict(zip(all_factor_names, model.coef_))
    # Convert the dictionary to a pandas DataFrame for a nice tabular display
    coefficient_df = pd.DataFrame(list(coefficient_dict.items()), columns=['Factor', 'Coefficient'])
    # Return the dataframe
    # Calculate the residuals
    residuals = stock_returns - model.predict(factor_loadings)
    # Calculate the square root of the residuals
    # Since residuals can be negative, we use the absolute value to ensure that the square root is a real number.
    sqrt_residuals = np.sqrt(np.abs(residuals))
    return coefficient_df, sqrt_residuals.sum()

def parameter_estimation_GLS(trade_date_rank):
    # Extracting risk factors
    risk_factor_names = ['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP', 'DASTD', 'CMRA', 'HSIGMA',
                         'EGRO', 'SGRO', 'BTOP', 'MLEV', 'DTOA', 'BLEV', 'STOM', 'STOQ', 'STOA']
    factor_loadings = []
    stock_returns = []
    for stock in normalized_results_array:
        # Extract factor loadings
        factors = [stock[risk_factor] for risk_factor in risk_factor_names]
        # Construct industry dummy variables
        industry_code_for_stock = industry_codes[industry_codes[:, 0] == stock['stock_code'], 1][0]
        industry_dummy_variables = [1 if industry_code == industry_code_for_stock else 0 for industry_code in
                                    map(str, range(1201, 1220))]
        # Add factor loadings and dummy variables to matrix
        factor_loadings.append(factors + industry_dummy_variables)
        # Calculate and store the return rates
        stock_data = all_stock_data_np[all_stock_data_np[:, 1] == stock['stock_code']]
        current_day_close_price = stock_data[stock_data[:, -1] == trade_date_rank, 21].astype(float)
        previous_day_close_price = stock_data[stock_data[:, -1] == trade_date_rank - 1, 21].astype(float)
        stock_return = current_day_close_price / previous_day_close_price - 1
        stock_returns.append(stock_return[0])
    factor_loadings = sm.add_constant(np.array(factor_loadings))
    stock_returns = np.array(stock_returns)
    # Now you can use the factor_loadings as X and stock_returns as y in your GLS model
    gls_model = sm.GLS(stock_returns, factor_loadings)
    gls_results = gls_model.fit()
    # Define all factor names
    all_factor_names = ['const'] + risk_factor_names + ['Ind_' + str(i) for i in range(1201, 1220)]
    # Create a dictionary that pairs each coefficient with its corresponding factor name
    coefficient_dict = dict(zip(all_factor_names, gls_results.params))
    # Convert the dictionary to a pandas DataFrame for a nice tabular display
    coefficient_df = pd.DataFrame(list(coefficient_dict.items()), columns=['Factor', 'Coefficient'])
    # Calculate the residuals
    residuals = gls_results.resid
    # Calculate the square root of the residuals
    # Since residuals can be negative, we use the absolute value to ensure that the square root is a real number.
    sqrt_residuals = np.sqrt(np.abs(residuals))
    return coefficient_df, sqrt_residuals.sum()

def calculate_covariance_matrix(normalized_results_array):
    # Extracting risk factors
    risk_factor_names = ['NLSIZE', 'Beta', 'RSTR', 'LNCAP', 'ETOP', 'CETOP', 'DASTD', 'CMRA', 'HSIGMA',
                         'EGRO', 'SGRO', 'BTOP', 'MLEV', 'DTOA', 'BLEV', 'STOM', 'STOQ', 'STOA']
    factor_loadings = []
    for stock in normalized_results_array:
        # Extract factor loadings
        factors = [stock[risk_factor] for risk_factor in risk_factor_names]
        # Add factor loadings to the list
        factor_loadings.append(factors)
    # Convert the list of factor loadings into a 2D numpy array
    factor_loadings_array = np.array(factor_loadings)
    # Compute the covariance matrix
    covariance_matrix = np.cov(factor_loadings_array, rowvar=False)
    return covariance_matrix


# ###########################################     Method 1     #####################################################
# # Define all factor names
# all_factor_names = risk_factor_names + ['Ind_' + str(i) for i in range(1201, 1220)]
# # Create a structured numpy array
# coefficients_array = np.zeros((len(model.coef_),), dtype=[('Factor', 'U10'), ('Coefficient', 'f4')])
# # Fill the array
# for i in range(len(model.coef_)):
#     coefficients_array[i] = (all_factor_names[i], model.coef_[i])
# # Print the structured numpy array
# print(coefficients_array)
# np.save('20230512_coefficients_array.npy', coefficients_array)

###########################################     Method 2     #####################################################
# # Convert the dictionary to a pandas DataFrame for a nice tabular display
# coefficient_df = pd.DataFrame(list(coefficient_dict.items()), columns=['Factor', 'Coefficient'])
# # Print the dataframe
# print(coefficient_df)
# np.save('20230512_coefficient_df.npy', coefficient_df)
# coefficient_dict = np.load('20230512_coefficient_df.npy', allow_pickle=True)


if __name__ == "__main__":
    trade_date_rank = 1228
    # OLS_coefficient_df, OLS_sqrt_residuals_sum = parameter_estimation_OLS(trade_date_rank)
    # print(f"The final coefficient_df result is \n {OLS_coefficient_df}")
    # print(f"The final sqrt_residuals_sum result is {OLS_sqrt_residuals_sum}")
    #
    # GLS_coefficient_df, GLS_sqrt_residuals_sum = parameter_estimation_GLS(trade_date_rank)
    # print(f"The final coefficient_df result is \n {GLS_coefficient_df}")
    # print(f"The final sqrt_residuals_sum result is {GLS_sqrt_residuals_sum}")

    covariance_matrix_20230512 = calculate_covariance_matrix(normalized_results_array)
    print(len(normalized_results_array))

    # print("Covariance matrix on date 20230512:")
    # print(covariance_matrix_20230512)

    # beta_value = result.loc[result['Factor'] == 'Beta', 'Coefficient'].values[0]
    # print('Beta:', beta_value)








