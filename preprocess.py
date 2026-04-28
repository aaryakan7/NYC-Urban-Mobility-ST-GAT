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
    
    # Process all Yellow Taxi files
    print("Loading and combining Yellow Taxi Data...")
    yellow_dfs = []
    for file in yellow_files:
        df = pd.read_parquet(file)
        df = df[(df['passenger_count'] > 0) & (df['trip_distance'] > 0)]
        df = df[['tpep_pickup_datetime', 'PULocationID']].copy()
        df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime'}, inplace=True)
        yellow_dfs.append(df)
        
    df_yellow = pd.concat(yellow_dfs, ignore_index=True) if yellow_dfs else pd.DataFrame()

    # Process all FHV files
    print("Loading and combining High Volume FHV (Uber/Lyft) Data...")
    fhv_dfs = []
    for file in fhv_files:
        df = pd.read_parquet(file)
        if 'pickup_datetime' in df.columns and 'PULocationID' in df.columns:
            df = df[['pickup_datetime', 'PULocationID']].copy()
            fhv_dfs.append(df)
            
    df_fhv = pd.concat(fhv_dfs, ignore_index=True) if fhv_dfs else pd.DataFrame()

    # Combine all months and floor to 15-minute intervals
    print("Aggregating into 15-minute intervals...")
    df_combined = pd.concat([df_yellow, df_fhv], ignore_index=True)
    df_combined['time_bin'] = df_combined['pickup_datetime'].dt.floor('15min')

    # Aggregate demand per zone per 15-min window
    demand_df = df_combined.groupby(['time_bin', 'PULocationID']).size().reset_index(name='demand')

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
