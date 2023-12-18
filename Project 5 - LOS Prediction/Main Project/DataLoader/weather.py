import pandas as pd
import matplotlib.pyplot as plt


df_rain = pd.read_csv('weather_data.csv')
    
def plot_precipitation_counts(df):
    """
    Plots a bar chart of the counts of different precipitation types.

    Parameters:
    - dataframe: A pandas DataFrame containing weather data.

    Returns:
    A bar plot showing the counts of different precipitation types.
    """
    count_series = df['preciptype'].value_counts()

    # Plotting
    plt.figure(figsize=(6, 4))
    count_series.plot(kind='bar')
    plt.title('Counts of Precipitation Types')
    plt.xlabel('Precipitation Type')
    plt.ylabel('Counts')
    plt.show()
    
    
def plot_rainy_hours_per_day(df):
    """
    Takes a DataFrame with a 'datetime' and 'preciptype' columns,
    converts 'datetime' to a date, filters for rainy 'preciptype',
    and plots the number of rainy hours per day.

    Parameters:
    - df: A pandas DataFrame with 'datetime' and 'preciptype' columns.

    Returns:
    A bar plot showing the number of rainy hours per day.
    """
    # Convert 'datetime' to datetime object and extract the date
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    # Filter for rows where 'preciptype' indicates rain
    rain_data = df[df['preciptype'].str.contains('rain', na=False)]

    # Group by date and count the number of rainy hours per day
    rainy_hours_per_day = rain_data.groupby('date').size()

    # Plotting
    plt.figure(figsize=(12, 6))
    rainy_hours_per_day.plot(kind='bar')
    plt.title('Number of Rainy Hours per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Rainy Hours')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    


def plot_total_rainfall_per_day(df):
    """
    Takes a DataFrame with 'datetime', 'preciptype', and 'precip' columns,
    converts 'datetime' to a date, filters for rainy 'preciptype',
    sums the 'precip' values for each day, and plots the total rainfall per day.

    Parameters:
    - df: A pandas DataFrame with 'datetime', 'preciptype', and 'precip' columns.

    Returns:
    A bar plot showing the total rainfall per day in millimeters.
    """
    # Convert 'datetime' to datetime object and extract the date
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    # Filter for rows where 'preciptype' indicates rain
    rain_data = df[df['preciptype'].str.contains('rain', na=False)]

    # Group by date and sum the 'precip' values for each day
    total_rain_per_day = rain_data.groupby('date')['precip'].sum()

    # Plotting
    plt.figure(figsize=(12, 6))
    total_rain_per_day.plot(kind='bar')
    plt.title('Total Rainfall per Day')
    plt.xlabel('Date')
    plt.ylabel('Total Rainfall (mm)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
def categorize_rainfall(amount):
    """
    Categorizes rainfall amounts into distinct intensity levels.
    This function takes a numerical value representing rainfall amount and 
    categorizes it into one of four intensity levels based on predefined 
    thresholds: 'Slight rain' for amounts less than 0.5 mm, 'Moderate rain' 
    for amounts from 0.5 mm up to 4.0 mm, 'Heavy rain' for amounts from 4.0 mm 
    up to 8.0 mm, and 'Very heavy rain' for amounts equal to or greater than 8.0 mm.

    Parameters:
    - amount: A numerical value representing the amount of rainfall.

    Returns:
    A string representing the categorized rainfall intensity.
    """
    if amount < 0.5:
        return 'Slight rain'
    elif 0.5 <= amount < 4.0:
        return 'Moderate rain'
    elif 4.0 <= amount < 8.0:
        return 'Heavy rain'
    else:  # Assuming that any value above or equal to 8.0 is 'Very heavy rain'
        return 'Very heavy rain'



def apply_rain_intensity_categorization(df):
    """
    Categorizes the amount of rainfall in a DataFrame and adds a new column with the categories.

    Parameters:
    - df: pandas DataFrame containing the precipitation data.
    """
    df['rain_intensity'] = df['precip'].apply(categorize_rainfall)
    


def plot_rain_intensity_distribution(df):
    """
    Plots the distribution of rain intensity levels in the dataset.

    This function counts the occurrences of each rain intensity level and 
    creates a bar plot to visualize the distribution.

    Parameters:
    - df: pandas DataFrame containing the rain intensity data.

    Returns:
    A bar plot showing the number of occurrences of each rain intensity level.
    """
    # Count the number of occurrences for each rain intensity category
    rain_intensity_counts_total = df['rain_intensity'].value_counts()

    # Plotting the counts of different rain intensities
    plt.figure(figsize=(10, 6))
    rain_intensity_counts_total.plot(kind='bar')

    # Setting the plot title and labels
    plt.title('Number of Hours by Rain Intensity')
    plt.xlabel('Rain Intensity')
    plt.ylabel('Number of Hours')
    plt.xticks(rotation=0)

    # Display the plot
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.show()


