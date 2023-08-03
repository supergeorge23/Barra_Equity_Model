import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Remove weekend data
def remove_consecutive_zeros(data):
    processed_data = []
    skip_next = False
    for i in range(len(data) - 1):  # iterate to the second-last element
        if skip_next:  # if flagged to skip, reset flag and continue
            skip_next = False
            continue
        if data[i] == 0 and data[i + 1] == 0:  # if current and next elements are both zero
            skip_next = True  # flag to skip next element
            continue
        processed_data.append(data[i])
    if not skip_next:  # append last element if not flagged to skip
        processed_data.append(data[-1])
    return processed_data


data = compute_factor_matrix_normalized_20230512 = np.load('multiple_date_portfolio_optimization_outcome_2.npy', allow_pickle=True)
# data[np.isnan(data)] = 0
# # using the function
processed_data = remove_consecutive_zeros(data)
# print(f"processed_data is be like: {processed_data}")
# # Calculate statistics
mean = np.mean(processed_data)
std_dev = np.std(processed_data)
median = np.median(processed_data)
q25 = np.percentile(processed_data, 25)
q75 = np.percentile(processed_data, 75)
cumulative_returns = (1 + np.array(processed_data)).cumprod()
cumulative_return_last_day = np.product(processed_data) - 1
max_return = np.maximum.accumulate(cumulative_returns)
drawdown = 1 - cumulative_returns / max_return
maximum_drawdown = np.max(drawdown)
# Define table data
table_data = [
    ["Mean", mean],
    ["Standard Deviation", std_dev],
    ["Median", median],
    ["25th Percentile", q25],
    ["75th Percentile", q75],
    # ['cumulative_returns', cumulative_returns],
    # ['max_return', max_return],
    # ['drawdown', drawdown],
    # ['maximum_drawdown', maximum_drawdown]
]

print(table_data)




# print(f"The length of data is {len(processed_data)}, the sum of the data is {sum(processed_data)}")
# assuming your data is stored in the list 'data'
# Plot as before
x = range(1, len(processed_data) + 1)  # generate X values
fig, ax = plt.subplots(figsize=(10, 10))  # adjust the height to make room for the table
ax.plot(x, processed_data)
ax.set_xlabel('Trade Date Rank')
ax.set_ylabel('Portfolio Return')
ax.set_title('Backtest from May 2022 to May 2023')
ax.grid(True)
# find the index of maximum and minimum, add 1 because x values start from 1
max_idx = np.argmax(processed_data) + 1
min_idx = np.argmin(processed_data) + 1
end_idx = len(processed_data)
# plot the maximum, minimum and end values
ax.plot(max_idx, processed_data[max_idx-1], 'ro')
ax.plot(min_idx, processed_data[min_idx-1], 'go')
ax.plot(end_idx, processed_data[end_idx-1], 'bo')
# annotate maximum, minimum and end values
ax.annotate(f'Max: {processed_data[max_idx-1]:.2f}', (max_idx, processed_data[max_idx-1]), textcoords="offset points", xytext=(-10,10), ha='center', color='red')
ax.annotate(f'Min: {processed_data[min_idx-1]:.2f}', (min_idx, processed_data[min_idx-1]), textcoords="offset points", xytext=(-10,-15), ha='center', color='green')
ax.annotate(f'End: {processed_data[end_idx-1]:.2f}', (end_idx, processed_data[end_idx-1]), textcoords="offset points", xytext=(-10,10), ha='center', color='blue')
table = plt.table(cellText=table_data, loc='bottom', bbox=[0.0, -0.35, 1.0, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.subplots_adjust(left=0.2, bottom=0.4)  # adjust bottom margin to make room for the table
plt.show()
# Save the plot
fig.savefig('portfolio_optimization_outcome.png', bbox_inches='tight')





multiple_date_portfolio_optimization_outcome_2 = np.load('portfolio_optimization_outcome_20230502.npy', allow_pickle=True)
print(f"multiple_date_portfolio_optimization_outcome_2 is \n {multiple_date_portfolio_optimization_outcome_2} "
      f"\n and its length is {len(multiple_date_portfolio_optimization_outcome_2)}"
      f"\n and its sum is {np.sum(multiple_date_portfolio_optimization_outcome_2)}")
# First, convert your returns to multiplicative factors
processed_data_factors = [1 + x / 100. for x in processed_data]
# Calculate the cumulative product of these factors
new_data = np.cumprod(processed_data_factors)
# Generate x values
x = range(1, len(new_data) + 1)
# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, new_data)
plt.xlabel('Trade Date Rank')
plt.ylabel('Compound Portfolio Return')
plt.title('Backtest from May 2022 to May 2023')
plt.grid(True)
plt.show()
plt.savefig('portfolio_optimization_outcome_2.png', bbox_inches='tight')


