# NYC Urban Mobility ST-GAT

A deep learning pipeline leveraging Spatio-Temporal Graph Attention Networks (ST-GAT) to forecast ride-hailing demand across 263 New York City Taxi and Limousine Commission (TLC) zones. By fusing historical mobility data with real-time weather and engineered cyclical time features, this model achieves a **15.58% weighted mean average percentage error** on unseen June 2024 demand data.

This project was developed for the CS 439 final project at Rutgers University.

---

## Model Architecture

Traditional time-series forecasting treats geographic zones as isolated environments. This architecture models NYC as a unified, breathing entity by separating spatial and temporal learning:

1. **Spatial Layer (GATv2):** A Graph Attention Network (v2) with 4 attention heads dynamically learns which neighboring zones are driving traffic at any given moment, rather than relying on static geographic borders.
2. **Temporal Layer (GRU):** A Gated Recurrent Unit processes a sliding window of 12 hours (`seq_len=48` at 15-minute intervals) to track historical trends and sequence logic.
3. **Exogenous Features:** The network ingests 14 distinct features, including cyclical sine/cosine encodings for time-of-day, apparent temperature, and boolean flags for rush hour, nightlife, and major holidays.

---

## Key Results

The model was trained on Jan-May 2024 data and strictly evaluated on unseen June 2024 data to prevent temporal data leakage. 

* **Overall WMAPE:** 15.58%
* **Rush Hour WMAPE:** 14.44%
* **Baseline Improvement:** >39% reduction in MSE versus a naive persistence baseline
* **Ablation Study:** Permutation feature importance revealed that **Apparent Temperature** and **Temperature (2m)** are the heaviest exogenous drivers of urban mobility, followed closely by precipitation and snow.

---

## Project Structure

* **`preprocess.py`**: The ETL engine. Ingests raw Parquet files (Yellow Taxi & High-Volume FHV), executes 15-minute aggregations, engineers temporal/weather features, builds the binary spatial adjacency matrix via `geopandas`, and saves the scaled dataset.
* **`models/model.py`**: The blueprint. Defines the PyTorch `STGAT` class, handling the complex tensor reshaping required to pass sequence windows through the GATv2 spatial layers and into the recurrent GRU layers.
* **`train.py`**: The training loop. Implements dynamic sliding-window data loaders, Automatic Mixed Precision (AMP) for GPU acceleration, Huber Loss for outlier resistance, and a StepLR learning rate scheduler.
* **`evaluate.ipynb`**: The diagnostics suite. Reverses the data scaling to calculate real-world metrics (WMAPE, MAE/RMSE), isolates the top 1% worst predictions for anomaly analysis, and generates geospatial heatmaps and feature importance charts.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/aaryakan7/NYC-Urban-Mobility-ST-GAT.git](https://github.com/aaryakan7/NYC-Urban-Mobility-ST-GAT.git)
   cd NYC-Urban-Mobility-ST-GAT
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
   *(Ensure you have `torch`, `torch-geometric`, `pandas`, `geopandas`, `scikit-learn`, `matplotlib`, and `tqdm` installed).*

3. **Data Requirements:**
   You will need to place the following data into a `data/` directory:
   * NYC TLC Trip Record Parquet files (Yellow Taxi & FHV) in `data/trips/`
   * TLC Zone Shapefiles (`taxi_zones.shp`) in `data/taxi_zones/`
   * Open-Meteo historical weather data CSV in `data/weather/`

---

## Execution Pipeline

To run the project from start to finish, execute the pipeline in the following order:

**1. Build the Dataset**
```bash
python preprocess.py
```
*Outputs: `final_stgat_input.csv`, `adjacency_matrix.npy`, `scaler.pkl`*

**2. Train the Model**
```bash
python train.py
```
*Outputs: `stgat_weights_best.pth` inside the `models/` directory.*

**3. Evaluate & Visualize**
Open `evaluate.ipynb` in Jupyter Notebook or VS Code, select your virtual environment kernel, and click **Run All** to generate the final diagnostic charts and accuracy metrics.
