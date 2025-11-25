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

def process_csv(file_path, data_type='deposits', mineral=None):
    """Process a CSV file: read, save to appropriate directory."""
    try:
        validate_file_integrity(file_path, 'csv')

        # Read CSV
        df = pd.read_csv(file_path)

        filename = os.path.basename(file_path)
        # Ensure filename ends with .csv if it doesn't (though it should based on check above)
        if not filename.endswith('.csv'):
             filename += '.csv'
             
        if data_type == 'features':
            # For features, we might save as parquet if it has geometry, or keep as CSV/parquet if not
            # save_geoparquet handles the logic of checking for lat/lon
            save_geoparquet(df, filename.replace('.csv', '.parquet'), 'data/features/', mineral)
        else:
            # For deposits, save_deposit_data handles checking for lat/lon
            # If it's a CSV without lat/lon, it will be saved as CSV
            save_deposit_data(df, filename, 'data/deposits/', mineral)

        return f"Successfully processed CSV: {filename}"

    except Exception as e:
        return f"Error processing CSV {file_path}: {str(e)}"

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