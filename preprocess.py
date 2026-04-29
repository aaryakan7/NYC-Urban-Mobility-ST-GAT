import pandas as pd
import geopandas as gpd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler

def process_trip_data(yellow_pattern, fhv_pattern):
    # Uses glob to find all files matching the pattern
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

    # Combine the already-aggregated dataframes
    print("Combining aggregated data...")
    if not aggregated_dfs:
        return pd.DataFrame()
        
    df_combined = pd.concat(aggregated_dfs, ignore_index=True)

    # Group one last time to sum overlapping bins (e.g., FHV and Yellow in the same 15min block)
    print("Finalizing 15-minute interval demand matrix...")
    demand_df = df_combined.groupby(['time_bin', 'PULocationID'])['demand'].sum().reset_index()

    # Pivot to have zones as columns and time bins as rows
    demand_matrix = demand_df.pivot(index='time_bin', columns='PULocationID', values='demand').fillna(0)
    return demand_matrix

def build_spatial_graph(shapefile_path):
    # Creates a binary adjacency matrix based on touching TLC zones
    print("Building spatial adjacency matrix...")
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.sort_values('LocationID').reset_index(drop=True)

    num_zones = len(gdf)
    adj_matrix = np.zeros((num_zones, num_zones))

    for i in range(num_zones):
        for j in range(num_zones):
            if i != j and gdf.geometry[i].intersects(gdf.iloc[j].geometry):
                adj_matrix[i, j] = 1

    # Save the matrix for the PyTorch model
    np.save('processed/adjacency_matrix.npy', adj_matrix)
    return adj_matrix, gdf['LocationID'].values

def integrate_weather_and_scale(demand_matrix, weather_path):
    print("Integrating Open-Meteo weather data...")
    df_weather = pd.read_csv(weather_path, skiprows=2)
    
    # Convert the 'time' column to a pandas datetime index
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather.set_index('time', inplace=True)
    
    # Resample weather to 15-minute intervals and forward-fill missing values
    df_weather_15m = df_weather.resample('15min').ffill()
    
    # Merge weather with trip demand data on the time index
    merged_df = demand_matrix.join(df_weather_15m, how='left').ffill().bfill()
    
    # Sine/Cosine Cyclic Transformations
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
    final_df.to_csv('processed/final_stgat_input.csv')
    return final_df

if __name__ == "__main__":
    # Use wildcards (*) to select all 2024 and 2025 parquet files
    yellow_pattern = 'data/trips/yellow_tripdata_*.parquet'
    fhv_pattern = 'data/trips/fhvhv_tripdata_*.parquet'
    
    demand = process_trip_data(yellow_pattern, fhv_pattern)
    adj_matrix, zone_ids = build_spatial_graph('data/taxi_zones/taxi_zones.shp')
    final_dataset = integrate_weather_and_scale(demand, 'data/weather/new_york_weather.csv')
    print("Preprocessing Complete. Data saved to /processed.")
