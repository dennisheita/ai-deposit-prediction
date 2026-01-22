import requests
import tempfile
import os
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

# API endpoint
API_URL = "http://localhost:8000"

def create_test_prediction_data(n_samples=5):
    """Create test prediction data with random coordinates"""
    # Create some random points near a gold deposit location
    lats = [47.5 + i * 0.1 for i in range(n_samples)]
    lons = [-120.5 + i * 0.1 for i in range(n_samples)]
    
    # Create features (random values based on training data ranges)
    data = {
        'lat': lats,
        'lon': lons,
        'elevation': [1000 + i * 100 for i in range(n_samples)],
        'slope': [5 + i * 2 for i in range(n_samples)],
        'aspect': [90 + i * 45 for i in range(n_samples)]
    }
    
    # Add geology codes (random)
    for i in range(1, 29):
        data[f'geology_code_{i}'] = [0] * n_samples
    data['geology_code_unknown'] = [0] * n_samples
    
    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
    
    return gdf

def test_prediction():
    """Test prediction endpoint with test data"""
    print("Creating test data...")
    gdf = create_test_prediction_data(5)
    
    # Save to CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        gdf.to_csv(tmp.name, index=False)
        test_file_path = tmp.name
    
    print(f"Test file created: {test_file_path}")
    
    try:
        # Test prediction with Copper mineral
        print("\nTesting prediction with Copper mineral...")
        files = {'file': open(test_file_path, 'rb')}
        data = {'mineral': 'Copper', 'threshold': '0.3'}
        
        response = requests.post(f"{API_URL}/predict", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Prediction map generated")
            if 'filename' in result:
                print(f"   Output filename: {result['filename']}")
            if 'map_html' in result:
                print(f"   Map HTML length: {len(result['map_html'])} characters")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
        
        # Test prediction with Gold mineral
        print("\nTesting prediction with Gold mineral...")
        files = {'file': open(test_file_path, 'rb')}
        data = {'mineral': 'Gold', 'threshold': '0.3'}
        
        response = requests.post(f"{API_URL}/predict", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Prediction map generated")
            if 'filename' in result:
                print(f"   Output filename: {result['filename']}")
            if 'map_html' in result:
                print(f"   Map HTML length: {len(result['map_html'])} characters")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
        
        # Test prediction with All Minerals
        print("\nTesting prediction with All Minerals...")
        files = {'file': open(test_file_path, 'rb')}
        data = {'mineral': 'All Minerals', 'threshold': '0.3'}
        
        response = requests.post(f"{API_URL}/predict", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Prediction map generated")
            if 'filename' in result:
                print(f"   Output filename: {result['filename']}")
            if 'map_html' in result:
                print(f"   Map HTML length: {len(result['map_html'])} characters")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    finally:
        # Cleanup
        os.unlink(test_file_path)

if __name__ == "__main__":
    print("="*50)
    print("AI Deposit Prediction API Test")
    print("="*50)
    
    # Test API connectivity
    print("\nTesting API connectivity...")
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            print(f"✅ API is running. Found {len(response.json()['models'])} models")
        else:
            print(f"❌ API connection failed: {response.status_code}")
            exit(1)
    except Exception as e:
        print(f"❌ Connection error: {e}")
        exit(1)
    
    # Test prediction endpoint
    test_prediction()
    
    print("\n" + "="*50)
    print("Test completed")
    print("="*50)
