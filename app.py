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
    "Download Center",
    "Data Management"
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

            # Check for continuous learning
            if st.session_state.get('continuous_learning', False):
                st.info("🔄 Continuous learning enabled - retraining model with new data...")
                with st.spinner("Retraining model..."):
                    # Get latest data for retraining
                    features_options = [f[1] for f in get_files() if f[3] == 'geoparquet']
                    deposits_options = [f[1] for f in get_files() if f[3] == 'deposit']

                    if features_options and deposits_options:
                        # Use the most recently uploaded files
                        features_file = features_options[-1]  # Last uploaded
                        deposits_file = deposits_options[-1]  # Last uploaded

                        try:
                            result = run_training_pipeline(features_file, deposits_file)
                            st.success(f"✅ Model retrained successfully! {result}")
                        except Exception as e:
                            st.warning(f"⚠️ Retraining failed: {str(e)}. You can manually retrain in the Training section.")
                    else:
                        st.warning("⚠️ Could not retrain - missing features or deposits data.")

            # Clean up temp files
            for path in file_paths:
                os.unlink(path)

# Training Section
elif page == "Training":
    st.title("Training Controls")
    st.write("Start training runs for deposit prediction models.")

    # Continuous Learning Mode
    st.subheader("🚀 Continuous Learning Mode")
    st.write("Automatically retrain the model when new data is uploaded.")

    continuous_learning = st.checkbox("Enable Continuous Learning", value=st.session_state.get('continuous_learning', False))
    st.session_state['continuous_learning'] = continuous_learning
    if continuous_learning:
        st.info("✅ Continuous learning enabled. The model will be retrained automatically when you upload new data in the Data Upload section.")
        st.write("**How it works:**")
        st.write("1. Upload new features or deposits data")
        st.write("2. The system automatically retrains using all available data")
        st.write("3. New model replaces the old one")
        st.write("4. Check Statistics Dashboard for updated performance")

    # Get all available files that could be used for training
    features_options = [f[1] for f in get_files() if f[3] == 'geoparquet']
    deposits_options = [f[1] for f in get_files() if f[3] == 'deposit']

    # Combine options for the main training file selection
    all_training_options = list(set(deposits_options + features_options))

    # Always show the button to train on existing pre-split data
    st.subheader("Quick Training")

    if continuous_learning:
        st.info("🔄 Continuous Learning is enabled. Each training improves the model with accumulated knowledge.")

        training_count = st.session_state.get('training_iterations', 0)
        if training_count > 0:
            st.write(f"📊 Completed {training_count} training iterations")

        button_text = "🚀 Start/Continue Continuous Training" if training_count == 0 else "🔄 Continue Training (Next Iteration)"
        if st.button(button_text):
            training_count += 1
            st.session_state['training_iterations'] = training_count

            with st.spinner(f"Training iteration #{training_count}..."):
                # Run the train_model.py script
                import subprocess
                result = subprocess.run(['python3', 'train_model.py'], capture_output=True, text=True, cwd='.')

                if result.returncode == 0:
                    st.success(f"✅ Training iteration #{training_count} completed successfully!")

                    # Extract and display metrics
                    lines = result.stdout.split('\n')
                    auc_line = next((line for line in lines if 'Test AUC:' in line), None)
                    acc_line = next((line for line in lines if 'Test Accuracy:' in line), None)

                    if auc_line:
                        auc = auc_line.split('Test AUC:')[1].strip()
                        st.metric("Test AUC", auc)
                    if acc_line:
                        acc = acc_line.split('Test Accuracy:')[1].strip()
                        st.metric("Test Accuracy", acc)

                    st.info("🎯 Model improved! Click 'Continue Training' for the next iteration.")
                else:
                    st.error(f"❌ Training iteration #{training_count} failed!")
                    st.code(result.stderr)

        if st.button("🔴 Stop Continuous Training"):
            st.session_state['training_iterations'] = 0
            st.success("Continuous training stopped. You can restart anytime.")
    else:
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
        deposits_file = training_file

        if st.button("Start Training on Uploaded Data"):
            with st.spinner("Training in progress..."):
                # If features_file is None, pass deposits_file as features_file too
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
    st.write("Run multiple training jobs with different hyperparameters to find the best model configuration.")

    st.subheader("What are Hyperparameters?")
    st.write("""
    **Hyperparameters** are settings that control how the machine learning algorithm learns:
    - **n_estimators**: Number of decision trees in the Random Forest (more = better but slower)
    - **max_depth**: Maximum depth of each tree (deeper = more complex patterns but risk of overfitting)
    - **min_samples_split**: Minimum samples needed to split a node
    - **min_samples_leaf**: Minimum samples in a leaf node

    Batch processing tests different combinations to find the optimal settings for your data.
    """)

    # Check if data is available
    features_options = [f[1] for f in get_files() if f[3] == 'geoparquet' and 'features' in f[4]]
    deposits_options = [f[1] for f in get_files() if f[3] == 'deposit']

    if not features_options or not deposits_options:
        st.error("No training data available. Please upload features and deposits data first.")
        st.info("💡 Tip: Upload your geological feature data (elevation, slope, aspect, geology codes) and known deposit locations.")
    else:
        st.subheader("Select Training Data")
        features_file = st.selectbox("Features file", features_options, key="batch_features")
        deposits_file = st.selectbox("Deposits file", deposits_options, key="batch_deposits")

        st.subheader("Hyperparameter Tuning")
        st.write("Select which parameters to test. The system will try all combinations.")

        col1, col2 = st.columns(2)

        with col1:
            n_estimators_options = st.multiselect(
                "n_estimators (trees in forest)",
                [50, 100, 200, 500],
                default=[100, 200]
            )
            max_depth_options = st.multiselect(
                "max_depth (tree depth)",
                [10, 20, 30, None],
                default=[20, None]
            )

        with col2:
            min_samples_split_options = st.multiselect(
                "min_samples_split",
                [2, 5, 10],
                default=[2, 5]
            )
            min_samples_leaf_options = st.multiselect(
                "min_samples_leaf",
                [1, 2, 4],
                default=[1, 2]
            )

        if st.button("Start Hyperparameter Tuning"):
            if not n_estimators_options or not max_depth_options:
                st.error("Please select at least one option for n_estimators and max_depth.")
            else:
                # Generate parameter combinations
                param_combinations = []
                for n_est in n_estimators_options:
                    for max_d in max_depth_options:
                        for min_split in min_samples_split_options:
                            for min_leaf in min_samples_leaf_options:
                                param_combinations.append({
                                    'n_estimators': [n_est],
                                    'max_depth': [max_d],
                                    'min_samples_split': [min_split],
                                    'min_samples_leaf': [min_leaf]
                                })

                st.write(f"🔄 Will test {len(param_combinations)} parameter combinations")
                st.write("This may take several minutes depending on your data size and parameter combinations.")

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, params in enumerate(param_combinations):
                    param_desc = f"n_est={params['n_estimators'][0]}, depth={params['max_depth'][0] or 'None'}, split={params['min_samples_split'][0]}, leaf={params['min_samples_leaf'][0]}"
                    status_text.text(f"Training model {i+1}/{len(param_combinations)}: {param_desc}")
                    try:
                        result = run_training_pipeline(features_file, deposits_file, param_grid=params)
                        # Extract AUC from result string
                        auc_match = result.split('AUC ') if 'AUC ' in result else ['N/A']
                        auc = auc_match[-1].split()[0] if len(auc_match) > 1 else 'N/A'
                        results.append(f"✅ Model {i+1}: {param_desc} - AUC: {auc}")
                    except Exception as e:
                        results.append(f"❌ Model {i+1}: {param_desc} - Failed: {str(e)[:50]}...")

                    progress_bar.progress((i+1) / len(param_combinations))

                status_text.text("🎉 Hyperparameter tuning completed!")
                st.subheader("Results Summary")
                st.write("Models trained with different hyperparameters:")

                # Show best result
                successful_results = [r for r in results if 'AUC:' in r]
                if successful_results:
                    best_result = max(successful_results, key=lambda x: float(x.split('AUC: ')[1]) if 'AUC: ' in x else 0)
                    st.success(f"🏆 Best performing model: {best_result}")

                # Show all results
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

# Data Management Section
elif page == "Data Management":
    st.title("Data Management")
    st.write("Manage uploaded datasets, models, and system data.")

    st.subheader("Delete All Datasets")
    st.warning("⚠️ This will permanently delete all uploaded files, trained models, predictions, and database records. This action cannot be undone.")

    if st.button("Delete All Datasets", type="primary"):
        with st.spinner("Deleting all datasets..."):
            import os
            import shutil
            import sqlite3

            # Delete data files
            data_dirs = ['data/features', 'data/deposits', 'data/predictions', 'models', 'logs']
            for dir_path in data_dirs:
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            st.warning(f"Could not delete {file_path}: {e}")

            # Reset database
            try:
                db_path = 'data/metadata.db'
                if os.path.exists(db_path):
                    os.unlink(db_path)
                # Recreate database
                from src.data_architecture import initialize_database
                initialize_database()
            except Exception as e:
                st.error(f"Could not reset database: {e}")

        st.success("All datasets have been deleted and the system has been reset.")
        st.info("The application will continue running. You can now upload new data.")