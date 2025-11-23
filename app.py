import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import tempfile
import zipfile
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
import threading
import time

# Import modules
from src.data_architecture import get_files, get_models, load_geoparquet, save_prediction_data
from src.data_ingestion import ingest_files
from src.feature_engineering import FeatureEngineer
from src.training_pipeline import run_training_pipeline
from src.monitoring import get_active_alerts, generate_performance_report, plot_performance_trends, plot_feature_importance_evolution
from src.prediction import run_prediction_pipeline

# Set page config
st.set_page_config(page_title="AI Deposit Prediction", layout="wide")

# Ensure data directories exist
os.makedirs('data/features', exist_ok=True)
os.makedirs('data/deposits', exist_ok=True)
os.makedirs('data/predictions', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section", [
    "Data Upload",
    "Training",
    "Statistics Dashboard",
    "Prediction",
    "Map Visualization",
    "Batch Processing",
    "Model Comparison",
    "Download Center"
])

# Data Upload Section
if page == "Data Upload":
    st.title("Data Upload")
    st.write("Upload shapefiles or CSV files for features or deposits.")

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['shp', 'zip', 'csv'])

    if uploaded_files:
        if st.button("Process Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            file_paths = []
            for i, file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
                    tmp.write(file.read())
                    file_paths.append(tmp.name)
                progress_bar.progress((i+1) / len(uploaded_files))

            results = ingest_files(file_paths)
            status_text.text("Processing complete!")
            st.write("Results:")
            for result in results:
                st.write(result)

            # Clean up temp files
            for path in file_paths:
                os.unlink(path)

# Training Section
elif page == "Training":
    st.title("Training Controls")
    st.write("Start training runs for deposit prediction models.")

    features_options = [f[1] for f in get_files() if f[3] == 'geoparquet' and 'features' in f[5]]
    deposits_options = [f[1] for f in get_files() if f[3] == 'deposit']

    if not features_options:
        st.error("No features files available. Please upload features data first.")
    elif not deposits_options:
        st.error("No deposits files available. Please upload deposits data first.")
    else:
        features_file = st.selectbox("Select features file", features_options)
        deposits_file = st.selectbox("Select deposits file", deposits_options)

        if st.button("Start Training"):
            with st.spinner("Training in progress..."):
                result = run_training_pipeline(features_file, deposits_file)
            st.success(result)

            # Display progress (simplified, as real-time is complex in Streamlit)
            st.write("Training completed. Check Statistics Dashboard for details.")

# Statistics Dashboard
elif page == "Statistics Dashboard":
    st.title("Statistics Dashboard")

    # Alerts
    alerts = get_active_alerts()
    if alerts:
        st.subheader("Active Alerts")
        for alert in alerts:
            severity_color = {'warning': 'orange', 'error': 'red', 'critical': 'darkred', 'info': 'blue'}
            st.markdown(f"<span style='color:{severity_color.get(alert[3], 'black')}; font-weight:bold;'>{alert[3].upper()}: {alert[2]}</span>", unsafe_allow_html=True)
            if st.button(f"Resolve Alert {alert[0]}", key=f"resolve_{alert[0]}"):
                from src.monitoring import resolve_alert
                resolve_alert(alert[0])
                st.rerun()

    models = get_models()
    st.write(f"Total Training Runs: {len(models)}")

    if models:
        df = pd.DataFrame(models, columns=['id', 'version', 'performance_metrics', 'created_date'])
        df['performance_metrics'] = df['performance_metrics'].apply(lambda x: json.loads(x) if x else {})
        st.dataframe(df)

        # Best metrics
        best_model = max(models, key=lambda x: json.loads(x[2])['best_score'] if x[2] else 0)
        st.write(f"Best Model: {best_model[1]}, Score: {json.loads(best_model[2])['best_score']}")

        # Performance report
        st.subheader("Performance Report")
        report = generate_performance_report()
        st.text(report)

        # Performance trends plot
        st.subheader("Performance Trends")
        fig_trends = plot_performance_trends()
        if fig_trends:
            st.pyplot(fig_trends)
        else:
            st.write("Not enough data for trends.")

        # Feature importance evolution
        st.subheader("Feature Importance Evolution")
        fig_fi = plot_feature_importance_evolution()
        if fig_fi:
            st.pyplot(fig_fi)
        else:
            st.write("No feature importance data available.")

        # Feature importance plot (if available)
        # Assuming we can load the model and get importances
        model_path = json.loads(best_model[2]).get('model_path')
        if model_path and os.path.exists(model_path):
            model = joblib.load(model_path)
            if hasattr(model, 'feature_importances_'):
                fig, ax = plt.subplots()
                ax.bar(range(len(model.feature_importances_)), model.feature_importances_)
                ax.set_title("Feature Importances")
                st.pyplot(fig)

# Prediction Section
elif page == "Prediction":
    st.title("Prediction Interface")
    st.write("Upload prediction areas and generate outputs.")

    prediction_file = st.file_uploader("Upload prediction shapefile or GeoJSON", type=['shp', 'zip', 'geojson'])
    model_version = st.selectbox("Select model", [m[1] for m in get_models()])
    threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Run Prediction") and prediction_file and model_version:
        with st.spinner("Running prediction pipeline..."):
            # Process uploaded file
            file_type = 'geojson' if prediction_file.name.endswith('.geojson') else 'shp'
            if file_type == 'shp' and prediction_file.name.endswith('.zip'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(prediction_file.read())
                    zip_path = tmp.name

                extract_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                shp_path = None
                for file in os.listdir(extract_dir):
                    if file.endswith('.shp'):
                        shp_path = os.path.join(extract_dir, file)
                        break
                prediction_area_path = shp_path
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
                    tmp.write(prediction_file.read())
                    prediction_area_path = tmp.name

            if prediction_area_path:
                try:
                    output_filename, pred_gdf = run_prediction_pipeline(prediction_area_path, model_version, threshold)
                    st.success(f"Prediction completed and saved as {output_filename}")
                    st.write(f"Generated {len(pred_gdf)} predictions")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

            # Clean up
            if 'zip_path' in locals():
                os.unlink(zip_path)
            if 'extract_dir' in locals():
                import shutil
                shutil.rmtree(extract_dir)
            if 'prediction_area_path' in locals() and os.path.exists(prediction_area_path):
                os.unlink(prediction_area_path)

# Map Visualization
elif page == "Map Visualization":
    st.title("Map Visualization")
    st.write("Interactive deposit probability heatmaps with overlays.")

    prediction_files = [f[1] for f in get_files() if f[3] == 'prediction']
    selected_pred = st.selectbox("Select prediction file", prediction_files)

    # Options for overlays
    show_heatmaps = st.checkbox("Show Probability Heatmap", value=True)
    show_predictions = st.checkbox("Show Prediction Points", value=False)
    overlay_faults = st.checkbox("Overlay Faults", value=False)
    overlay_rivers = st.checkbox("Overlay Rivers", value=False)
    overlay_deposits = st.checkbox("Overlay Known Deposits", value=False)

    if selected_pred:
        pred_gdf = gpd.read_file(f"data/predictions/{selected_pred}")

        # Create Folium map
        center_lat = pred_gdf.geometry.centroid.y.mean()
        center_lon = pred_gdf.geometry.centroid.x.mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Add heatmap
        if show_heatmaps:
            heat_data = [[row.geometry.y, row.geometry.x, row.probability] for idx, row in pred_gdf.iterrows()]
            HeatMap(heat_data, name="Probability Heatmap").add_to(m)

        # Add prediction points
        if show_predictions:
            for idx, row in pred_gdf.iterrows():
                color = 'red' if row['prediction'] == 1 else 'blue'
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"Prob: {row.probability:.2f}, Pred: {row['prediction']}, Conf: {row.get('confidence', 'N/A'):.2f}"
                ).add_to(m)

        # Overlay faults
        if overlay_faults:
            try:
                fault_files = [f for f in get_files() if 'fault' in f[1] and f[3] == 'geoparquet']
                if fault_files:
                    faults_gdf = load_geoparquet(fault_files[0][1], 'data/features/')
                    folium.GeoJson(faults_gdf.to_json(), name="Faults").add_to(m)
                else:
                    st.warning("No fault files available")
            except Exception as e:
                st.warning(f"Faults layer error: {str(e)}")

        # Overlay rivers
        if overlay_rivers:
            try:
                river_files = [f for f in get_files() if 'river' in f[1] and f[3] == 'geoparquet']
                if river_files:
                    rivers_gdf = load_geoparquet(river_files[0][1], 'data/features/')
                    folium.GeoJson(rivers_gdf.to_json(), name="Rivers").add_to(m)
                else:
                    st.warning("No river files available")
            except Exception as e:
                st.warning(f"Rivers layer error: {str(e)}")

        # Overlay deposits
        if overlay_deposits:
            try:
                deposit_files = [f for f in get_files() if f[3] == 'deposit']
                if deposit_files:
                    deposits_path = deposit_files[0][5]
                    deposits_gdf = gpd.read_file(deposits_path)
                    for idx, row in deposits_gdf.iterrows():
                        folium.Marker(
                            location=[row.geometry.y, row.geometry.x],
                            popup="Known Deposit",
                            icon=folium.Icon(color='green', icon='info-sign')
                        ).add_to(m)
                else:
                    st.warning("No deposit files available")
            except Exception as e:
                st.warning(f"Deposits layer error: {str(e)}")

        # Add layer control
        folium.LayerControl().add_to(m)

        st_folium(m, width=700, height=500)

# Batch Processing
elif page == "Batch Processing":
    st.title("Batch Processing")
    st.write("Queue multiple training jobs.")

    # Simple implementation: allow multiple parameter sets
    st.write("Feature: Select multiple parameter combinations.")

    # For simplicity, just run multiple trainings sequentially
    if st.button("Start Batch Training"):
        # Example: run with different n_estimators
        params_list = [
            {'n_estimators': [100]},
            {'n_estimators': [500]},
            {'n_estimators': [1000]}
        ]

        results = []
        progress_bar = st.progress(0)
        for i, params in enumerate(params_list):
            result = run_training_pipeline('features.parquet', 'deposits.parquet', param_grid=params)
            results.append(result)
            progress_bar.progress((i+1) / len(params_list))

        st.write("Batch results:")
        for res in results:
            st.write(res)

# Model Comparison
elif page == "Model Comparison":
    st.title("Model Comparison")

    models = get_models()
    model1 = st.selectbox("Select Model 1", [m[1] for m in models])
    model2 = st.selectbox("Select Model 2", [m[1] for m in models if m[1] != model1])

    if model1 and model2:
        m1_data = next(m for m in models if m[1] == model1)
        m2_data = next(m for m in models if m[1] == model2)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Model {model1}")
            metrics1 = json.loads(m1_data[2]) if m1_data[2] else {}
            st.write(f"Best Score: {metrics1.get('best_score', 'N/A')}")
            st.write(f"Best Params: {metrics1.get('best_params', 'N/A')}")

        with col2:
            st.subheader(f"Model {model2}")
            metrics2 = json.loads(m2_data[2]) if m2_data[2] else {}
            st.write(f"Best Score: {metrics2.get('best_score', 'N/A')}")
            st.write(f"Best Params: {metrics2.get('best_params', 'N/A')}")

# Download Center
elif page == "Download Center":
    st.title("Download Center")

    files = get_files()
    df = pd.DataFrame(files, columns=['id', 'filename', 'type', 'upload_date', 'status', 'path'])
    st.dataframe(df)

    selected_file = st.selectbox("Select file to download", df['filename'].tolist())

    if selected_file:
        file_row = df[df['filename'] == selected_file].iloc[0]
        file_path = file_row['path']
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                st.download_button("Download", f, file_name=selected_file)
        else:
            st.error("File not found")