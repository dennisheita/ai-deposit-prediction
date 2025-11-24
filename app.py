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
from src.prediction import run_prediction_pipeline, convert_spatial_to_csv

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
    st.write("Upload shapefiles or CSV files. The system will automatically detect if they are features or deposits data.")

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

    # Get all available files that could be used for training
    # Features might be in 'geoparquet' (features) or 'deposit' (if it's a combined CSV)
    features_options = [f[1] for f in get_files() if f[3] == 'geoparquet']
    deposits_options = [f[1] for f in get_files() if f[3] == 'deposit']
    
    # Combine options for the main training file selection since the user might have uploaded a single CSV
    # that got classified as 'deposit' but contains everything
    all_training_options = list(set(deposits_options + features_options))

    # Always show the button to train on existing pre-split data
    st.subheader("Quick Training")
    if st.button("Train on Existing Pre-split Data"):
        with st.spinner("Training on existing data..."):
            # Run the train_model.py script
            import subprocess
            result = subprocess.run(['python3', 'train_model.py'], capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                st.success("Model trained successfully on existing data!")
                st.text("Output:")
                st.code(result.stdout)
            else:
                st.error("Training failed!")
                st.text("Error:")
                st.code(result.stderr)
        st.write("Training completed. Check Statistics Dashboard for details.")

    st.subheader("Training on Uploaded Data")
    # Allow training if we have any data
    if not all_training_options:
        st.error("No uploaded training data available. Please upload data first or use the quick training option above.")
    else:
        st.write("Select data for training. If you uploaded a single CSV with features and labels, select it here.")

        # Main training file (can be deposits or combined)
        training_file = st.selectbox("Select training data file", all_training_options)

        # Optional separate features file (only if needed)
        features_file = st.selectbox("Select separate features file (optional)", ["None"] + features_options)
        if features_file == "None":
            features_file = None

        # If features_file is not selected, we assume training_file has everything or is the deposits file
        # We pass training_file as deposits_file to the pipeline
        deposits_file = training_file

        if st.button("Start Training on Uploaded Data"):
            with st.spinner("Training in progress..."):
                # If features_file is None, pass deposits_file as features_file too
                # The pipeline handles the logic of checking if it's a combined file
                feat_file_to_pass = features_file if features_file else deposits_file

                result = run_training_pipeline(feat_file_to_pass, deposits_file)
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
    st.write("Upload prediction data with the same features as training (CSV format) or shapefiles/GeoJSON for spatial areas.")

    # Conversion tool
    st.subheader("Convert Spatial Data to CSV")
    st.write("Convert shapefiles or GeoJSON to CSV format with lat/lon coordinates for prediction.")
    convert_file = st.file_uploader("Upload shapefile or GeoJSON to convert", type=['shp', 'zip', 'geojson'], key='convert')
    if convert_file and st.button("Convert to CSV"):
        with st.spinner("Converting..."):
            # Save uploaded file temporarily
            if convert_file.name.endswith('.zip'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(convert_file.read())
                    temp_path = tmp.name
                # Extract
                extract_dir = tempfile.mkdtemp()
                import zipfile
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                # Find shapefile
                for file in os.listdir(extract_dir):
                    if file.endswith('.shp'):
                        input_path = os.path.join(extract_dir, file)
                        break
                else:
                    st.error("No shapefile found in zip")
                    input_path = None
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{convert_file.name.split('.')[-1]}") as tmp:
                    tmp.write(convert_file.read())
                    input_path = tmp.name

            if input_path:
                output_path = convert_spatial_to_csv(input_path)
                st.success(f"Converted and saved as {os.path.basename(output_path)}")
                # Offer download
                with open(output_path, "rb") as f:
                    st.download_button("Download Converted CSV", f, file_name=os.path.basename(output_path))

        # Clean up
        if 'temp_path' in locals():
            os.unlink(temp_path)
        if 'extract_dir' in locals():
            import shutil
            shutil.rmtree(extract_dir)
        if 'input_path' in locals() and os.path.exists(input_path):
            os.unlink(input_path)

    st.subheader("Run Prediction")
    prediction_file = st.file_uploader("Upload prediction data (CSV, shapefile, or GeoJSON)", type=['csv', 'shp', 'zip', 'geojson'])
    model_version = st.selectbox("Select model", [m[1] for m in get_models()])
    threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Run Prediction") and prediction_file and model_version:
        with st.spinner("Running prediction pipeline..."):
            # Determine file type
            if prediction_file.name.endswith('.csv'):
                file_type = 'csv'
                suffix = '.csv'
            elif prediction_file.name.endswith('.geojson'):
                file_type = 'geojson'
                suffix = '.geojson'
            else:
                file_type = 'shp'
                suffix = '.zip' if prediction_file.name.endswith('.zip') else '.shp'

            # Process uploaded file
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
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(prediction_file.read())
                    prediction_area_path = tmp.name

            if prediction_area_path:
                try:
                    output_filename, pred_gdf = run_prediction_pipeline(prediction_area_path, model_version, threshold)
                    st.success(f"Prediction completed and saved as {output_filename}")
                    st.write(f"Generated {len(pred_gdf)} predictions")

                    # Provide download link for the shapefile
                    shapefile_path = f"data/predictions/{output_filename}"
                    if os.path.exists(shapefile_path):
                        # Create a zip file with all shapefile components
                        import zipfile
                        zip_path = shapefile_path.replace('.shp', '.zip')
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                                file_path = shapefile_path.replace('.shp', ext)
                                if os.path.exists(file_path):
                                    zipf.write(file_path, os.path.basename(file_path))

                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="Download Prediction Shapefile",
                                data=f,
                                file_name=f"{output_filename.replace('.shp', '.zip')}",
                                mime="application/zip"
                            )
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

    # Get prediction files from database
    prediction_files_db = [f[1] for f in get_files() if f[3] == 'prediction']

    # Also check for shapefiles in the predictions directory
    import glob
    prediction_dir = 'data/predictions/'
    shapefiles = glob.glob(os.path.join(prediction_dir, '*.shp'))
    prediction_files_dir = [os.path.basename(f).replace('.shp', '') for f in shapefiles]

    # Combine and deduplicate
    prediction_files = list(set(prediction_files_db + prediction_files_dir))

    if not prediction_files:
        st.warning("No prediction files available. Please run a prediction first.")
        selected_pred = None
    else:
        selected_pred = st.selectbox("Select prediction file", prediction_files)

    # Options for overlays
    show_heatmaps = st.checkbox("Show Probability Heatmap", value=True)
    show_predictions = st.checkbox("Show Prediction Points", value=False)
    overlay_faults = st.checkbox("Overlay Faults", value=False)
    overlay_rivers = st.checkbox("Overlay Rivers", value=False)
    overlay_deposits = st.checkbox("Overlay Known Deposits", value=False)

    if selected_pred:
        # Try with .shp extension first
        shp_path = f"data/predictions/{selected_pred}.shp"
        if os.path.exists(shp_path):
            pred_gdf = gpd.read_file(shp_path)
        else:
            # Try without extension (might be a different format)
            pred_gdf = gpd.read_file(f"data/predictions/{selected_pred}")

        # Create Folium map
        center_lat = pred_gdf.geometry.centroid.y.mean()
        center_lon = pred_gdf.geometry.centroid.x.mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Get correct column names (shapefiles truncate to 10 chars)
        prob_col = 'probabilit' if 'probabilit' in pred_gdf.columns else 'probability'
        pred_col = 'predictio' if 'predictio' in pred_gdf.columns else 'prediction'
        conf_col = 'confidenc' if 'confidenc' in pred_gdf.columns else 'confidence'

        # Add heatmap
        if show_heatmaps:
            heat_data = [[row.geometry.y, row.geometry.x, row[prob_col]] for idx, row in pred_gdf.iterrows()]
            HeatMap(heat_data, name="Probability Heatmap").add_to(m)

        # Add prediction points
        if show_predictions:
            for idx, row in pred_gdf.iterrows():
                color = 'red' if row[pred_col] == 1 else 'blue'
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"Prob: {row[prob_col]:.2f}, Pred: {row[pred_col]}, Conf: {row.get(conf_col, 'N/A'):.2f}"
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
    st.write("Run multiple training jobs with different parameters.")

    st.write("This will train models with different n_estimators values on the existing pre-split data.")

    if st.button("Start Batch Training"):
        # Run batch training with different n_estimators
        n_estimators_list = [100, 500, 1000]

        results = []
        progress_bar = st.progress(0)
        for i, n_est in enumerate(n_estimators_list):
            st.write(f"Training with n_estimators={n_est}...")
            import subprocess
            result = subprocess.run(['python3', 'train_model.py', str(n_est)], capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                results.append(f"n_estimators={n_est}: Success - {result.stdout.split('Test AUC:')[1].split()[0] if 'Test AUC:' in result.stdout else 'Completed'}")
            else:
                results.append(f"n_estimators={n_est}: Failed")
            progress_bar.progress((i+1) / len(n_estimators_list))

        st.write("Batch results:")
        for res in results:
            st.write(res)
        st.success("Batch training completed!")

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