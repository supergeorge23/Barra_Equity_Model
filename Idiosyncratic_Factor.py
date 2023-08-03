# Idiosyncratic_Factor.py
import matplotlib
matplotlib.use('Agg')  # Set backend to 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_daily_idiosyncratic_volatility(residuals, lambda_, h, t):
    # First, calculate the mean of residuals
    residuals_mean = np.mean(residuals)
    # Compute the EWMA
    sum_numerator = sum([lambda_**(t - s) * (residuals[s] - residuals_mean)**2 for s in range(t - h, t + 1)])
    sum_denominator = sum([lambda_**(t - s) for s in range(t - h, t + 1)])
    daily_volatility = sum_numerator / sum_denominator
    return daily_volatility
def calculate_all_daily_volatilities(residuals_list, lambda_, h, t):
    daily_volatilities = [calculate_daily_idiosyncratic_volatility(residuals.flatten(), lambda_, h, t)
                          for residuals in residuals_list]
    return daily_volatilities
def plot_daily_volatilities(daily_volatilities):
    plt.figure(figsize=(10, 6))
    # If you have the corresponding dates, replace range(len(daily_volatilities)) with your dates
    plt.plot(range(len(daily_volatilities)), daily_volatilities, marker='o', linestyle='-')
    plt.title("Daily Volatilities")
    plt.xlabel("Trading Date Rank")
    plt.ylabel("Daily Volatility")
    plt.grid(True)
    # Save the figure to a file instead of trying to show it
    plt.savefig('daily_volatilities.png')







# sqrt_residuals_df_1 = np.random.rand(1, 5264)
# sqrt_residuals_df_2 = np.random.rand(1, 5264)
# sqrt_residuals_df_3 = np.random.rand(1, 5264)
# sqrt_residuals_df_4 = np.random.rand(1, 5264)
# sqrt_residuals_df_5 = np.random.rand(1, 5264)
# sqrt_residuals_df_6 = np.random.rand(1, 5264)
# sqrt_residuals_df_7 = np.random.rand(1, 5264)
# sqrt_residuals_df_8 = np.random.rand(1, 5264)
# sqrt_residuals_df_9 = np.random.rand(1, 5264)
# sqrt_residuals_df_10 = np.random.rand(1, 5264)
#
# residuals_list = [sqrt_residuals_df_1, sqrt_residuals_df_2, sqrt_residuals_df_3, sqrt_residuals_df_4, sqrt_residuals_df_5,
#                   sqrt_residuals_df_6, sqrt_residuals_df_7, sqrt_residuals_df_8, sqrt_residuals_df_9, sqrt_residuals_df_10]

residuals_list = np.load('full_sqrt_residuals_sum_list.npy', allow_pickle=True)

lambda_ = 0.94
h = 1 #22 # Adjust the value of h according to your context
t = 2 #30 # Adjust the value of t according to your context


daily_volatilities = calculate_all_daily_volatilities(residuals_list, lambda_, h, t)

for i, daily_volatility in enumerate(daily_volatilities, 1):
    print(f"The daily idiosyncratic volatility for sqrt_residuals_df_{i} is {daily_volatility}")

print(f"daily_volatilities is be like: {daily_volatilities}")

# # Example of calculating daily idiosyncratic volatility for sqrt_residuals_df_1
# daily_volatility_1 = calculate_daily_idiosyncratic_volatility(sqrt_residuals_df_1.flatten(), lambda_, h, t)
#
# print(f"The daily idiosyncratic volatility for sqrt_residuals_df_1 is {daily_volatility_1}")
# call the function
plot_daily_volatilities(daily_volatilities)
plt.savefig('volatility_plot.png')


