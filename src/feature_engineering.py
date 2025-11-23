import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from .data_architecture import load_geoparquet, save_geoparquet
import os

class FeatureEngineer:
    def __init__(self, deposits_file='deposits.parquet', features_dir='data/features/'):
        self.deposits_file = deposits_file
        self.features_dir = features_dir
        self.deposits_gdf = None
        self.base_gdf = None

    def load_data(self, base_filename):
        """Load base GeoDataFrame and deposits data."""
        self.base_gdf = load_geoparquet(base_filename, self.features_dir)
        try:
            self.deposits_gdf = load_geoparquet(self.deposits_file, 'data/deposits/')
        except FileNotFoundError:
            print("Deposits file not found, some features may not be computed.")

    def compute_spatial_features(self, gdf):
        """Compute spatial features: distance to deposits, spatial autocorrelation, kernel density."""
        features = gpd.GeoDataFrame(index=gdf.index, geometry=gdf.geometry)

        # Distance to nearest deposit
        if self.deposits_gdf is not None:
            features['distance_to_deposit'] = gdf.geometry.apply(
                lambda geom: self.deposits_gdf.distance(geom).min()
            )

        # Kernel density estimation (simplified, assuming 2D points)
        coords = np.array([[geom.x, geom.y] for geom in gdf.geometry])
        kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde.fit(coords)
        features['kernel_density'] = kde.score_samples(coords)

        # Spatial autocorrelation (Moran's I approximation using distance weights)
        # Simplified: average distance to k nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        features['spatial_autocorr'] = np.mean(distances, axis=1)

        return features

    def compute_topographic_features(self, gdf):
        """Compute topographic features: elevation, slope, aspect, curvature, hillshade."""
        features = gpd.GeoDataFrame(index=gdf.index, geometry=gdf.geometry)

        # Assume elevation is a column in gdf
        if 'elevation' in gdf.columns:
            features['elevation'] = gdf['elevation']

            # Slope: approximate using neighboring elevations (simplified)
            # For real implementation, use DEM derivatives
            # Here, assume grid and compute gradients
            # Placeholder: random for demo
            features['slope'] = np.random.uniform(0, 45, len(gdf))  # degrees

            features['aspect'] = np.random.uniform(0, 360, len(gdf))  # degrees
            features['curvature'] = np.random.uniform(-1, 1, len(gdf))
            features['hillshade'] = np.random.uniform(0, 255, len(gdf))  # 0-255

        return features

    def compute_geological_features(self, gdf):
        """Compute geological features: distance to faults, lithology."""
        features = gpd.GeoDataFrame(index=gdf.index, geometry=gdf.geometry)

        # Assume faults are in another layer
        faults_file = 'faults.parquet'
        try:
            faults_gdf = load_geoparquet(faults_file, self.features_dir)
            features['distance_to_fault'] = gdf.geometry.apply(
                lambda geom: faults_gdf.distance(geom).min()
            )
        except FileNotFoundError:
            features['distance_to_fault'] = np.nan

        # Lithology: assume categorical column
        if 'lithology' in gdf.columns:
            features['lithology'] = gdf['lithology']

        return features

    def compute_hydrological_features(self, gdf):
        """Compute hydrological features: distance to rivers, watersheds."""
        features = gpd.GeoDataFrame(index=gdf.index, geometry=gdf.geometry)

        # Rivers
        rivers_file = 'rivers.parquet'
        try:
            rivers_gdf = load_geoparquet(rivers_file, self.features_dir)
            features['distance_to_river'] = gdf.geometry.apply(
                lambda geom: rivers_gdf.distance(geom).min()
            )
        except FileNotFoundError:
            features['distance_to_river'] = np.nan

        # Watersheds: assume polygon layer
        watersheds_file = 'watersheds.parquet'
        try:
            watersheds_gdf = load_geoparquet(watersheds_file, self.features_dir)
            features['watershed_id'] = gdf.geometry.apply(
                lambda geom: watersheds_gdf[watersheds_gdf.contains(geom)].index[0] if len(watersheds_gdf[watersheds_gdf.contains(geom)]) > 0 else -1
            )
        except FileNotFoundError:
            features['watershed_id'] = -1

        return features

    def compute_climatic_features(self, gdf):
        """Compute climatic features: precipitation, temperature patterns."""
        features = gpd.GeoDataFrame(index=gdf.index, geometry=gdf.geometry)

        # Assume climatic data as columns or interpolated
        if 'precipitation' in gdf.columns:
            features['precipitation'] = gdf['precipitation']
        if 'temperature' in gdf.columns:
            features['temperature'] = gdf['temperature']

        # Patterns: seasonal averages, etc. (simplified)
        # Assume monthly data, compute annual avg
        temp_cols = [col for col in gdf.columns if 'temp_' in col]
        if temp_cols:
            features['temp_avg'] = gdf[temp_cols].mean(axis=1)

        precip_cols = [col for col in gdf.columns if 'precip_' in col]
        if precip_cols:
            features['precip_avg'] = gdf[precip_cols].mean(axis=1)

        return features

    def compute_spatial_lag_features(self, gdf, features_df):
        """Compute spatial lag features."""
        # Lag of a feature, e.g., average of neighbors
        from sklearn.neighbors import kneighbors_graph
        coords = np.array([[geom.x, geom.y] for geom in gdf.geometry])
        graph = kneighbors_graph(coords, n_neighbors=5, mode='connectivity', include_self=False)
        # For each feature, compute lag
        lag_features = {}
        for col in features_df.select_dtypes(include=[np.number]).columns:
            lag = graph.dot(features_df[col].values) / graph.sum(axis=1).A1
            lag_features[f'{col}_lag'] = lag
        return pd.DataFrame(lag_features, index=gdf.index)

    def compute_multi_scale_features(self, gdf, features_df):
        """Compute multi-scale features: features at different buffer sizes."""
        scales = [100, 500, 1000]  # meters
        multi_features = {}
        for scale in scales:
            buffered = gdf.buffer(scale)
            # Simplified: count points within buffer (assume self)
            # For real, intersect with other layers
            multi_features[f'buffer_{scale}_count'] = [1] * len(gdf)  # placeholder
        return pd.DataFrame(multi_features, index=gdf.index)

    def compute_temporal_features(self, gdf):
        """Compute temporal features if time data available."""
        features = gpd.GeoDataFrame(index=gdf.index, geometry=gdf.geometry)
        if 'date' in gdf.columns:
            gdf['date'] = pd.to_datetime(gdf['date'])
            features['year'] = gdf['date'].dt.year
            features['month'] = gdf['date'].dt.month
            features['season'] = gdf['date'].dt.month % 12 // 3 + 1  # 1-4
        return features

    def process_and_save(self, base_filename, output_filename):
        """Main pipeline: load, compute all features, save."""
        self.load_data(base_filename)

        if self.base_gdf is None:
            raise ValueError("Base GeoDataFrame not loaded.")

        # Compute all feature categories
        spatial_feat = self.compute_spatial_features(self.base_gdf)
        topo_feat = self.compute_topographic_features(self.base_gdf)
        geo_feat = self.compute_geological_features(self.base_gdf)
        hydro_feat = self.compute_hydrological_features(self.base_gdf)
        clim_feat = self.compute_climatic_features(self.base_gdf)
        temp_feat = self.compute_temporal_features(self.base_gdf)

        # Combine
        all_features = pd.concat([spatial_feat.drop(columns='geometry'), topo_feat.drop(columns='geometry'),
                                  geo_feat.drop(columns='geometry'), hydro_feat.drop(columns='geometry'),
                                  clim_feat.drop(columns='geometry'), temp_feat.drop(columns='geometry')], axis=1)

        # Spatial lag and multi-scale on key features
        lag_feat = self.compute_spatial_lag_features(self.base_gdf, all_features[['elevation', 'distance_to_deposit']] if 'elevation' in all_features.columns else pd.DataFrame())
        multi_feat = self.compute_multi_scale_features(self.base_gdf, all_features)

        final_features = pd.concat([all_features, lag_feat, multi_feat], axis=1)

        # Merge back to GeoDataFrame
        result_gdf = self.base_gdf.copy()
        for col in final_features.columns:
            result_gdf[col] = final_features[col]

        # Save to feature store
        save_geoparquet(result_gdf, output_filename, self.features_dir)

        return f"Processed features saved to {output_filename}"