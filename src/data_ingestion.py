import os
import zipfile
import geopandas as gpd
import pandas as pd
from src.data_architecture import validate_crs, save_geoparquet, save_deposit_data, insert_file

def unzip_file(file_path, extract_to):
    """Unzip a file to the specified directory."""
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        raise ValueError(f"Failed to unzip {file_path}: {str(e)}")

def validate_file_integrity(file_path, file_type):
    """Validate file integrity based on type."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if file_type == 'shapefile':
        base_name = os.path.splitext(file_path)[0]
        required_extensions = ['.shp', '.shx', '.dbf']
        for ext in required_extensions:
            if not os.path.exists(base_name + ext):
                raise ValueError(f"Missing shapefile component: {base_name + ext}")
    elif file_type == 'csv':
        if not file_path.endswith('.csv'):
            raise ValueError(f"Invalid CSV file: {file_path}")

def process_shapefile(file_path, data_type='features', mineral=None):
    """Process a shapefile: unzip if needed, validate, convert to GeoParquet, save."""
    try:
        # Determine if zipped
        if file_path.endswith('.zip'):
            extract_dir = os.path.dirname(file_path)
            unzip_file(file_path, extract_dir)
            # Assume the shapefile is inside, find .shp
            for file in os.listdir(extract_dir):
                if file.endswith('.shp'):
                    shp_path = os.path.join(extract_dir, file)
                    break
            else:
                raise ValueError("No .shp file found in zip.")
        else:
            shp_path = file_path

        validate_file_integrity(shp_path, 'shapefile')

        # Read shapefile
        gdf = gpd.read_file(shp_path)

        # Convert to EPSG:4326 if not already
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')

        # Validate CRS
        validate_crs(gdf)

        # Spatial indexing
        gdf.sindex

        # Convert to GeoParquet and save
        if data_type == 'deposits':
            filename = os.path.basename(shp_path).replace('.shp', '.geojson')
            directory = 'data/deposits/'
            save_deposit_data(gdf, filename, directory, mineral)
        else:
            filename = os.path.basename(shp_path).replace('.shp', '.parquet')
            directory = 'data/features/'
            save_geoparquet(gdf, filename, directory, mineral)

        return f"Successfully processed shapefile: {filename}"

    except Exception as e:
        return f"Error processing shapefile {file_path}: {str(e)}"

def process_geojson(file_path, data_type='features', mineral=None):
    """Process a GeoJSON file: read, validate, save."""
    try:
        # Read GeoJSON
        gdf = gpd.read_file(file_path)

        # Set or convert CRS to EPSG:4326
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        elif gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')

        # Validate CRS
        validate_crs(gdf)

        # Spatial indexing
        gdf.sindex

        filename = os.path.basename(file_path)
        
        # Save based on data type
        if data_type == 'deposits':
            directory = 'data/deposits/'
            save_deposit_data(gdf, filename, directory, mineral)
        else:
            if filename.endswith('.geojson'):
                out_filename = filename.replace('.geojson', '.parquet')
            elif filename.endswith('.json'):
                out_filename = filename.replace('.json', '.parquet')
            else:
                 out_filename = filename + '.parquet'

            directory = 'data/features/'
            save_geoparquet(gdf, out_filename, directory, mineral)
            filename = out_filename

        return f"Successfully processed GeoJSON: {filename}"

    except Exception as e:
        return f"Error processing GeoJSON {file_path}: {str(e)}"

# ... (omitted process_csv) ...

def detect_data_type(file_path):
    """Detect if file is features or deposits based on content."""
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            if 'label' in df.columns:
                return 'deposits'
            else:
                return 'features'
        except Exception:
            return 'features'
    elif file_path.endswith('.geojson') or file_path.endswith('.json'):
        # For GeoJSON, check filename for hints
        if 'deposit' in file_path.lower():
            return 'deposits'
        else:
            return 'features'
    else:
        # For shapefiles, assume features unless filename contains 'deposit'
        if 'deposit' in file_path.lower():
            return 'deposits'
        else:
            return 'features'

def ingest_files(file_list, mineral=None):
    """Ingest a list of files, auto-detecting type and processing accordingly."""
    results = []
    for file_path in file_list:
        data_type = detect_data_type(file_path)
        if file_path.endswith('.shp') or file_path.endswith('.zip'):
            result = process_shapefile(file_path, data_type, mineral)
        elif file_path.endswith('.csv'):
            result = process_csv(file_path, data_type, mineral)
        elif file_path.endswith('.geojson') or file_path.endswith('.json'):
            result = process_geojson(file_path, data_type, mineral)
        else:
            result = f"Unsupported file type: {file_path}"
        results.append(result)
    return results

def download_prediction(filename):
    """Download a prediction file."""
    path = os.path.join('data/predictions/', filename)
    if os.path.exists(path):
        # In a real app, this would trigger a download, but here just return path
        return f"Download ready: {path}"
    else:
        raise FileNotFoundError(f"Prediction file {filename} not found.")
