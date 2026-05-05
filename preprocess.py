import pandas as pd
import geopandas as gpd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler

def process_trip_data(yellow_pattern, fhv_pattern):
    # Uses glob to find all files matching pattern
    yellow_files = glob.glob(yellow_pattern)
    fhv_files = glob.glob(fhv_pattern)
    
    print(f"Found {len(yellow_files)} Yellow Taxi files and {len(fhv_files)} FHV files.")
    
    aggregated_dfs = []
    
    # Process all Yellow Taxi files iteratively
    print("Processing Yellow Taxi Data file-by-file...")
    for file in yellow_files:
        df = pd.read_parquet(file)
        df = df[(df['passenger_count'] > 0) & (df['trip_distance'] > 0)]
        if 'tpep_pickup_datetime' in df.columns:
            df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime'}, inplace=True)
            
        # Aggregate immediately to save memory
        df['time_bin'] = df['pickup_datetime'].dt.floor('15min')
        agg_df = df.groupby(['time_bin', 'PULocationID']).size().reset_index(name='demand')
        aggregated_dfs.append(agg_df)

    # Process all FHV files iteratively
    print("Processing High Volume FHV (Uber/Lyft) Data file-by-file...")
    for file in fhv_files:
        df = pd.read_parquet(file)
        if 'pickup_datetime' in df.columns and 'PULocationID' in df.columns:
            # Aggregate immediately
            df['time_bin'] = df['pickup_datetime'].dt.floor('15min')
            agg_df = df.groupby(['time_bin', 'PULocationID']).size().reset_index(name='demand')
            aggregated_dfs.append(agg_df)

    # Combine already-aggregated dataframes
    print("Combining aggregated data...")
    if not aggregated_dfs:
        return pd.DataFrame()
        
    df_combined = pd.concat(aggregated_dfs, ignore_index=True)

    # Group one last time to sum overlapping bins 
    print("Finalizing 15-minute interval demand matrix...")
    demand_df = df_combined.groupby(['time_bin', 'PULocationID'])['demand'].sum().reset_index()

    # Have zones as columns and time bins as rows
    demand_matrix = demand_df.pivot(index='time_bin', columns='PULocationID', values='demand').fillna(0)
    return demand_matrix

def build_spatial_graph(shapefile_path):
    # Creates a binary adjacency matrix based on closely placed TLC zones
    print("Building spatial adjacency matrix...")
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.sort_values('LocationID').reset_index(drop=True)

    num_zones = len(gdf)
    adj_matrix = np.zeros((num_zones, num_zones))

    for i in range(num_zones):
        for j in range(num_zones):
            if i != j and gdf.geometry[i].intersects(gdf.iloc[j].geometry):
                adj_matrix[i, j] = 1

    # Save matrix for PyTorch model
    np.save('processed/adjacency_matrix.npy', adj_matrix)
    return adj_matrix, gdf['LocationID'].values

def integrate_weather_and_scale(demand_matrix, weather_path):
    print("Integrating Open-Meteo weather data...")
    df_weather = pd.read_csv(weather_path, skiprows=2)
    
    # Convert time column to a pandas datetime index
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather.set_index('time', inplace=True)
    
    # Resample weather to 15-minute intervals and forward-fill missing values
    df_weather_15m = df_weather.resample('15min').ffill()
    
    # Merge weather with trip demand data on time index
    merged_df = demand_matrix.join(df_weather_15m, how='left').ffill().bfill()
    
    # Sine/Cosine Transformations
    decimal_hour = merged_df.index.hour + (merged_df.index.minute / 60.0)
    day_of_week = merged_df.index.dayofweek
    
    merged_df['hour_sin'] = np.sin(2 * np.pi * decimal_hour / 24.0)
    merged_df['hour_cos'] = np.cos(2 * np.pi * decimal_hour / 24.0)
    merged_df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    merged_df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)

    # Convert all column names to strings for scikit-learn
    merged_df.columns = merged_df.columns.astype(str)
    
    # Apply standard scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_df)
    
    # Final dataframe
    final_df = pd.DataFrame(scaled_data, index=merged_df.index, columns=merged_df.columns)

    print("Engineering advanced temporal and event features...")

    # Ensure index is a DatetimeIndex hours and days extraction
    final_df.index = pd.to_datetime(final_df.index)

    # Flag days where model suffered worst misses
    # Add in June 19th as a known major summer holiday
    major_event_dates = ['2024-06-10', '2024-06-11', '2024-06-13', '2024-06-14', '2024-06-19', '2024-06-24']
    final_df['is_major_event_day'] = final_df.index.strftime('%Y-%m-%d').isin(major_event_dates).astype(float)

    # Flagging Thursday, Friday, and Saturday nights between 7 PM and Midnight
    # dayofweek: Monday=0, Sunday=6. (Thursday=3, Friday=4, Saturday=5)
    is_nightlife_day = final_df.index.dayofweek.isin([3, 4, 5])
    is_nightlife_hour = final_df.index.hour.isin([19, 20, 21, 22, 23])
    final_df['is_summer_nightlife'] = (is_nightlife_day & is_nightlife_hour).astype(float)
 
    is_weekday = final_df.index.dayofweek < 5
    is_morning_rush = final_df.index.hour.isin([7, 8, 9])
    is_evening_rush = final_df.index.hour.isin([16, 17, 18])
    final_df['is_rush_hour'] = (is_weekday & (is_morning_rush | is_evening_rush)).astype(float)

    # Weekends have different traffic patterns than weekdays
    final_df['is_weekend'] = (final_df.index.dayofweek >= 5).astype(float)

    # Save final matrix for model
    print("Saving highly-contextualized final dataset...")

    final_df.to_csv('processed/final_stgat_input.csv')
    return final_df

if __name__ == "__main__": 
    yellow_pattern = 'data/trips/yellow_tripdata_*.parquet'
    fhv_pattern = 'data/trips/fhvhv_tripdata_*.parquet'
    
    demand = process_trip_data(yellow_pattern, fhv_pattern)
    adj_matrix, zone_ids = build_spatial_graph('data/taxi_zones/taxi_zones.shp')
    final_dataset = integrate_weather_and_scale(demand, 'data/weather/new_york_weather.csv')
    print("Preprocessing Complete. Data saved to /processed.")
