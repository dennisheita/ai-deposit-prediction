from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import tempfile
import zipfile
from src.data_ingestion import ingest_files
from src.prediction import run_prediction_pipeline
from src.data_architecture import get_models, get_models_by_mineral, get_files
from src.monitoring import get_active_alerts, generate_performance_report, plot_performance_trends, plot_feature_importance_evolution
import matplotlib.pyplot as plt
import io
import base64
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this

# Ensure directories exist
os.makedirs('data/features', exist_ok=True)
os.makedirs('data/deposits', exist_ok=True)
os.makedirs('data/predictions', exist_ok=True)
os.makedirs('models', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            tmp.write(file.read())
            file_path = tmp.name
        # Ingest file
        results = ingest_files([file_path])
        os.unlink(file_path)
        flash('File uploaded and processed successfully')
        return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        mineral = request.form.get('mineral', 'All Minerals')
        threshold = float(request.form.get('threshold', 0.5))
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Process file
            if file.filename.endswith('.zip'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(file.read())
                    zip_path = tmp.name
                extract_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                shp_path = None
                for f in os.listdir(extract_dir):
                    if f.endswith('.shp'):
                        shp_path = os.path.join(extract_dir, f)
                        break
                prediction_area_path = shp_path
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
                    tmp.write(file.read())
                    prediction_area_path = tmp.name

            # Get model
            if mineral == 'All Minerals':
                models = get_models()
            else:
                models = get_models_by_mineral(mineral)
            if not models:
                flash('No models available')
                return redirect(url_for('predict'))
            model_version = models[-1][1]  # Latest model

            # Run prediction
            output_filename, pred_gdf = run_prediction_pipeline(prediction_area_path, model_version, threshold, mineral=mineral)

            # Create map
            center_lat = pred_gdf.geometry.centroid.y.mean()
            center_lon = pred_gdf.geometry.centroid.x.mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

            prob_col = 'probabilit' if 'probabilit' in pred_gdf.columns else 'probability'
            pred_col = 'predictio' if 'predictio' in pred_gdf.columns else 'prediction'

            if prob_col in pred_gdf.columns:
                heat_data = [[row.geometry.y, row.geometry.x, row[prob_col]] for idx, row in pred_gdf.iterrows()]
                HeatMap(heat_data, radius=15).add_to(m)

            if pred_col in pred_gdf.columns:
                for idx, row in pred_gdf.iterrows():
                    color = 'red' if row[pred_col] == 1 else 'blue'
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=6,
                        color=color,
                        fill=True,
                        fill_color=color,
                        popup=f"Probability: {row.get(prob_col, 'N/A'):.2f}<br>Prediction: {'Deposit' if row[pred_col] == 1 else 'No Deposit'}",
                        fill_opacity=0.7
                    ).add_to(m)

            folium.LayerControl().add_to(m)

            # Save map
            map_html = m._repr_html_()

            # Clean up
            if 'zip_path' in locals():
                os.unlink(zip_path)
            if 'extract_dir' in locals():
                import shutil
                shutil.rmtree(extract_dir)
            if 'prediction_area_path' in locals():
                os.unlink(prediction_area_path)

            return render_template('results.html', map_html=map_html, filename=output_filename)

    # GET request
    minerals = ["Copper", "Gold", "Uranium"]
    return render_template('predict.html', minerals=minerals)

@app.route('/map')
def map_view():
    # Get all prediction files
    prediction_files = [f for f in get_files() if f[3] == 'prediction']
    m = folium.Map(location=[0, 0], zoom_start=2)
    for f in prediction_files:
        mineral = f[6] or 'Unknown'
        shp_path = f"data/predictions/{f[1]}.shp"
        if os.path.exists(shp_path):
            pred_gdf = gpd.read_file(shp_path)
            prob_col = 'probabilit' if 'probabilit' in pred_gdf.columns else 'probability'
            pred_col = 'predictio' if 'predictio' in pred_gdf.columns else 'prediction'
            for idx, row in pred_gdf.iterrows():
                if row[pred_col] == 1:
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        color='red',
                        fill=True,
                        fill_color='red',
                        popup=f"Mineral: {mineral}<br>Probability: {row.get(prob_col, 'N/A'):.2f}",
                        fill_opacity=0.7
                    ).add_to(m)
    folium.LayerControl().add_to(m)
    map_html = m._repr_html_()
    return render_template('map.html', map_html=map_html)

@app.route('/stats')
def stats():
    mineral = request.args.get('mineral', 'All Minerals')
    if mineral == 'All Minerals':
        models = get_models()
    else:
        models = get_models_by_mineral(mineral)

    alerts = get_active_alerts()
    report = generate_performance_report()

    # Performance trends plot
    fig_trends = plot_performance_trends()
    trends_img = None
    if fig_trends:
        buf = io.BytesIO()
        fig_trends.savefig(buf, format='png')
        buf.seek(0)
        trends_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_trends)

    # Feature importance plot
    fig_fi = plot_feature_importance_evolution()
    fi_img = None
    if fig_fi:
        buf = io.BytesIO()
        fig_fi.savefig(buf, format='png')
        buf.seek(0)
        fi_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_fi)

    return render_template('stats.html', models=models, alerts=alerts, report=report, trends_img=trends_img, fi_img=fi_img, selected_mineral=mineral)

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    mineral = request.args.get('mineral', 'All Minerals')
    if mineral == 'All Minerals':
        models = get_models()
    else:
        models = get_models_by_mineral(mineral)

    model1 = None
    model2 = None
    if request.method == 'POST':
        model1_id = request.form.get('model1')
        model2_id = request.form.get('model2')
        model1 = next((m for m in models if str(m[0]) == model1_id), None)
        model2 = next((m for m in models if str(m[0]) == model2_id), None)

    return render_template('compare.html', models=models, model1=model1, model2=model2, selected_mineral=mineral)

@app.route('/download')
def download():
    mineral = request.args.get('mineral', 'All Minerals')
    if mineral == 'All Minerals':
        files = get_files()
    else:
        files = [f for f in get_files() if f[6] == mineral]
    return render_template('download.html', files=files, selected_mineral=mineral)

@app.route('/download/<int:file_id>')
def download_file(file_id):
    files = get_files()
    file_info = next((f for f in files if f[0] == file_id), None)
    if file_info and os.path.exists(file_info[5]):
        return send_from_directory(os.path.dirname(file_info[5]), os.path.basename(file_info[5]), as_attachment=True)
    flash('File not found')
    return redirect(url_for('download'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)