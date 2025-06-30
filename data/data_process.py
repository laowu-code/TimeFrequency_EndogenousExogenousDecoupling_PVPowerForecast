import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
# 保留指定日期范围内的数据
def filter_data_by_date_range(file_path, start_date, end_date, output_file_path):
    # 读取数据
    df = pd.read_csv(file_path)
    # 将第一列转换为日期格式以便筛选
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')
    # 筛选指定日期范围内的行
    filtered_df = df[(df[df.columns[0]] >= start_date) & (df[df.columns[0]] <= end_date)]
    # 将筛选后的数据保存到新的CSV文件
    filtered_df.to_csv(output_file_path, index=False)

def resample_data(input_file, output_file, original_freq='5T', target_freq='15T', fill_method='ffill', fill_value=0):
    # Load the data
    df = pd.read_csv(input_file)
    # Convert the first column to datetime
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')
    # Ensure the time column is set as the index
    df = df.set_index(df.columns[0])
    df = df.resample(original_freq).mean()
    # Handle missing values before resampling
    if fill_method:
        df = df.fillna(method=fill_method)
    # Resample to the target frequency and apply the adjustment (mean * 3)
    df_resampled = df.resample(target_freq).mean() * (int(target_freq[:-1]) // int(original_freq[:-1]))
    if df_resampled.isnull().any().any():
        print("数据框中存在NaN值")
    df_resampled = df_resampled.fillna(df_resampled.mean())
    print(df_resampled.mean())
    df_resampled = df_resampled.reset_index()
    df_resampled.to_csv(output_file, index=False)
    return df_resampled

def feature_plot(data,  test_size=0.2, random_state=None, save_heatmap_path=None):
    """
    Splits a dataset, computes a correlation matrix, and plots a lower triangular heatmap with correlation coefficients.

    Parameters:
    - data (pd.DataFrame): The input dataset as a Pandas DataFrame.
    - target_column (str): The name of the target column for prediction.
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int): Random seed for reproducibility (default is None).
    - save_heatmap_path (str): File path to save the heatmap (optional).

    Returns:
    - dict: A dictionary containing:
        - 'train_set': The training set as a DataFrame.
        - 'test_set': The test set as a DataFrame.
        - 'correlation_matrix': A correlation matrix as a DataFrame.
    """
    # Split the data into training and test sets
    train_set, test_set = train_test_split(data.iloc[:,1:], test_size=test_size, random_state=random_state)

    # Compute the correlation matrix
    correlation_matrix = train_set.corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        mask=mask,  # Apply the mask
        annot=True,  # Show the correlation coefficients
        fmt=".2f",  # Format for the annotations
        cmap="coolwarm",  # Color map
        cbar=True  # Show the color bar
    )
    plt.title('Lower Triangle Correlation Matrix')
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate column names
    plt.yticks(fontsize=10,rotation=45)  # Adjust row name font size
    # plt.title('Lower Triangle Correlation Matrix', fontsize=14)  # Add a title
    # Save the heatmap if a path is provided
    plt.savefig('../pic/data_heatmap.svg',format='SVG',dpi=600, bbox_inches='tight')
    plt.show()

    # Return the results
    return {
        'train_set': train_set,
        'test_set': test_set,
        'correlation_matrix': correlation_matrix
    }

def split_by_season(input_file, timestamp_column, output_prefix):
    """
    Splits a CSV file into four seasonal subsets based on timestamp's month.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - timestamp_column (str): Column name containing the timestamp.
    - output_prefix (str): Prefix for the output file names.

    Returns:
    - None: Saves four CSV files with seasonal data.
    """
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Ensure the timestamp column is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Extract the month from the timestamp
    df['month'] = df[timestamp_column].dt.month

    # Define the season groups
    seasons = {
        'Spring': [9, 10, 11],   # September to November
        'Summer': [12, 1, 2],  # December to February
        'Fall': [3, 4, 5],   # March to May
        'Winter': [6, 7, 8]    # June to August
    }

    # Iterate through each season and filter data
    for season, months in seasons.items():
        # Use modulo 12 arithmetic to handle months that span year boundaries (e.g., 12, 1, 2)
        subset = df[df['month'].isin(months)]
        output_file = f"../data/Site_1B/{output_prefix}_{season}.csv"

        # Save the subset to a new CSV file
        subset.drop(columns=['month'], inplace=True)  # Remove the helper 'month' column
        subset.to_csv(output_file, index=False)
        print(f"Saved {season} data to {output_file}")
# Example usage
# split_csv_by_season("data.csv", timestamp_column="timestamp", output_prefix="seasonal_data")
# split_by_season(file2, 'timestamp', '1B_15min_data_03_2020-02_2023')

def split_data_by_year(data, timestamp_column, H, P, target_column):
    """
    Splits data into training and testing sets for each year, ensuring data doesn't span across years.
    Uses the past H steps to predict the next P steps for each year.

    Parameters:
    - data (pd.DataFrame): The input dataset as a Pandas DataFrame.
    - timestamp_column (str): Column name containing the timestamp.
    - H (int): The number of historical steps to use for prediction.
    - P (int): The number of future steps to predict.
    - target_column (str): The column name to predict.

    Returns:
    - dict: A dictionary where the key is the year and the value is a list of tuples (X_train, y_train, X_test, y_test).
    """
    # Ensure timestamp is in datetime format
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])

    # Extract the year from the timestamp
    data['year'] = data[timestamp_column].dt.year

    # Dictionary to store results for each year
    year_splits = {}

    # Loop through each year and create the training/testing split
    for year in data['year'].unique():
        year_data = data[data['year'] == year]
        time_steps = len(year_data)

        year_splits[year] = []

        # Loop through each time step for the current year
        for i in range(H, time_steps - P + 1):
            train_data = year_data.iloc[i - H:i]  # Historical data (H steps)
            test_data = year_data.iloc[i:i + P]   # Future data (P steps)

            # Collect the features and target columns for both train and test
            X_train = train_data.drop(columns=[timestamp_column, target_column, 'year'])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[timestamp_column, target_column, 'year'])
            y_test = test_data[target_column]

            # Store the split data for this year
            year_splits[year].append((X_train, y_train, X_test, y_test))

    return year_splits

def split_continuous_samples(data, timestamp_column, H, P, target_column):
    """
    Splits the data into samples where time intervals are continuous, ensuring that the samples are only
    taken from contiguous time periods without any missing time steps.

    Parameters:
    - data (pd.DataFrame): The input dataset as a Pandas DataFrame.
    - timestamp_column (str): The column containing timestamps.
    - H (int): The number of historical time steps to use for prediction.
    - P (int): The number of future time steps to predict.
    - target_column (str): The column containing the target values.

    Returns:
    - list: A list of tuples (X_train, y_train, X_test, y_test) for each valid continuous time window.
    """
    # Ensure the data is sorted by the timestamp column
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    data = data.sort_values(by=timestamp_column)

    # Calculate time differences to find continuous periods
    data['time_diff'] = data[timestamp_column].diff().dt.total_seconds()

    # Identify the start of each continuous time block (when time_diff is not equal to 1 step)
    continuous_blocks = []
    start_idx = 0

    for i in range(1, len(data)):
        # If there is a gap (time difference more than one step), we mark the end of the current block
        if data['time_diff'].iloc[i] > 24 * 60 * 60:  # Example: gap > 24 hour (you can adjust the threshold)
            continuous_blocks.append((start_idx, i - 1))
            start_idx = i

    # Add the last block
    continuous_blocks.append((start_idx, len(data) - 1))

    # Store the results for valid continuous blocks
    valid_splits = []

    # Loop through each continuous block
    for start, end in continuous_blocks:
        block_data = data.iloc[start:end + 1]
        time_steps = len(block_data)

        # If there are enough time steps for splitting (H historical + P future)
        if time_steps > H + P:
            for i in range(H, time_steps - P + 1):
                train_data = block_data.iloc[i - H:i]  # Historical data (H steps)
                test_data = block_data.iloc[i:i + P]  # Future data (P steps)

                # Collect the features and target columns for both train and test
                X_train = train_data.drop(columns=[timestamp_column, target_column, 'time_diff'])
                y_train = train_data[target_column]
                X_test = test_data.drop(columns=[timestamp_column, target_column, 'time_diff'])
                y_test = test_data[target_column]

                # Store the split data
                valid_splits.append((X_train, y_train, X_test, y_test))

    return valid_splits




def interpolate_with_monthly_mean(input_file, output_folder, time_freq='1H', start_time=None, end_time=None):
    """
    对CSV文件中缺失的时间戳和数据使用当月对应时间点的平均值进行补全，并输出补全信息。
    只对给定时间段的数据进行处理。

    参数：
    - input_file (str): 输入CSV文件路径。
    - output_folder (str): 处理后文件的保存文件夹路径。
    - time_freq (str): 时间间隔频率，默认值为 '1H'（1小时）。
    - start_time (str or pd.Timestamp): 数据处理的起始时间（可选），格式：'YYYY-MM-DD HH:MM:SS'。
    - end_time (str or pd.Timestamp): 数据处理的结束时间（可选），格式：'YYYY-MM-DD HH:MM:SS'。

    返回：
    - output_file (str): 处理后文件的保存路径。
    - filled_count (int): 补全的样本点个数。
    - missing_time_ranges (list): 缺失的时间段范围列表。
    """
    # 检查输出文件夹是否存在，不存在则创建
    os.makedirs(output_folder, exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 确保第一列为时间戳，命名为 'timestamp'
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 设置时间戳为索引
    df.set_index('timestamp', inplace=True)

    # 如果提供了时间范围参数，则筛选出指定时间段的数据
    if start_time:
        start_time = pd.to_datetime(start_time)
        df = df[df.index >= start_time]
    if end_time:
        end_time = pd.to_datetime(end_time)
        df = df[df.index <= end_time]

    # 生成完整的时间范围
    time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=time_freq)

    # 重建DataFrame，确保时间序列完整（未出现的时间点填充为NaN）
    df = df.reindex(time_range)

    # 找到缺失的时间索引和时间段
    missing_indices = df[df.isna().any(axis=1)].index
    missing_time_ranges = []
    if len(missing_indices) > 0:
        # 分组计算连续时间段
        gaps = (missing_indices.to_series().diff() > pd.Timedelta(time_freq)).cumsum()
        grouped_ranges = missing_indices.to_series().groupby(gaps)
        missing_time_ranges = [
            (group.min(), group.max()) for _, group in grouped_ranges
        ]

    # 计算补全所需的月内平均值
    monthly_means = df.groupby([df.index.month, df.index.hour]).transform('mean')

    # 补全缺失值，使用月内对应时间点的平均值
    missing_before = df.isna().sum().sum()  # 缺失值总数（补全前）
    df.fillna(monthly_means, inplace=True)
    missing_after = df.isna().sum().sum()  # 缺失值总数（补全后）

    # 统计补全信息
    filled_count = missing_before - missing_after

    # 重置索引并保存
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timestamp'}, inplace=True)

    # 构造输出文件路径
    output_file = os.path.join(output_folder, 'monthly_mean_interpolated_file.csv')
    df.to_csv(output_file, index=False)

    print(f"文件已成功保存到: {output_file}")
    print(f"补全样本点个数: {filled_count}")
    for x in missing_time_ranges:
        print(f"补全的时间段: {x}")

    return output_file, filled_count, missing_time_ranges


def plot_correlation_matrix(input_file, output_image='correlation_matrix.png'):
    """
    读取CSV文件，计算各列之间的相关性，并绘制下三角相关性矩阵热力图。
    时间戳列将被排除在外，只有其他数据列会参与计算。

    参数：
    - input_file (str): 输入CSV文件路径。
    - output_image (str): 输出相关性矩阵图像的文件名，默认为 'correlation_matrix.png'。
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 确保第一列为时间戳，命名为 'timestamp'
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

    # 移除时间戳列，只保留数值列
    df_data = df.drop(columns=['timestamp'])

    # 计算相关性矩阵
    corr_matrix = df_data.corr()

    # 创建一个掩码，将相关性矩阵的上三角部分设置为True
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # 设置画布大小
    plt.figure(figsize=(10, 8))

    # 绘制相关性矩阵热力图
    sns.heatmap(corr_matrix,
                annot=True,          # 在热力图上标注相关系数数值
                fmt='.2f',          # 设置数值格式为2位小数
                cmap='coolwarm',    # 热力图的颜色映射
                mask=mask,          # 只显示下三角
                square=True,        # 确保矩阵是正方形的
                linewidths=0.5,     # 设置网格线宽度
                cbar_kws={"shrink": .8},  # 调整颜色条
                annot_kws={"size": 10})  # 设置数值字体大小

    # 保存图像
    plt.tight_layout()  # 自动调整布局
    # plt.savefig(output_image)
    plt.show()

    # print(f"相关性矩阵图已保存至 {output_image}")


# file='./site_1B/monthly_mean_interpolated_file.csv'
# plot_correlation_matrix(file)
# start_time='2021-03-01 00:00:00'
# end_time='2021-06-01 00:00:00'
# file='./site_1B/raw_data_03_2020-02_2023.csv'
# output_file, filled_count, time_range_filled = interpolate_with_monthly_mean(
#     file, './site_1B', time_freq='5min',start_time=start_time,end_time=end_time
# )
# start_time='2021-03-01 00:00:00'
# end_time='2021-06-01 00:00:00'
# file='./site_24/raw_site_24.csv'
# output_file, filled_count, time_range_filled = interpolate_with_monthly_mean(
#     file, './site_24', time_freq='5min',start_time=start_time,end_time=end_time
# )
#

# start_time='2018-06-30 16:00:00'
# end_time='2019-06-13 15:45:00'
file='./site_PVOD/station03.csv'
output_file, filled_count, time_range_filled = interpolate_with_monthly_mean(
    file, './site_PVOD', time_freq='15min',#start_time=start_time,end_time=end_time
)

