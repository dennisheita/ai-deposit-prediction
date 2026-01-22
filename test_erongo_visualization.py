#!/usr/bin/env python3
"""
Test the Erongo visualization with existing data.
"""

import geopandas as gpd
import folium
from folium.plugins import HeatMap
import os

def test_erongo_map():
    """Test creating an Erongo map with existing predictions."""
    
    print("Testing Erongo visualization...")
    
    # Load Erongo boundary
    erongo_gdf = gpd.read_file('data/erongo/Erongo.geojson')
    
    # Create map
    center_lat = erongo_gdf.geometry.centroid.y.mean()
    center_lon = erongo_gdf.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

    # Add Erongo boundary
    folium.GeoJson(
        erongo_gdf.to_json(),
        name='Erongo Boundary',
        style_function=lambda x: {
            'fillColor': '#ffffe0',
            'color': '#ff8c00',
            'weight': 3,
            'fillOpacity': 0.1
        }
    ).add_to(m)

    # Check if there are any Erongo predictions
    predictions_dir = 'data/predictions'
    erongo_pred_files = [f for f in os.listdir(predictions_dir) if 'erongo' in f.lower() and f.endswith('.shp')]
    
    if erongo_pred_files:
        print(f"Found {len(erongo_pred_files)} Erongo prediction files")
        
        for pred_file in erongo_pred_files:
            try:
                pred_gdf = gpd.read_file(os.path.join(predictions_dir, pred_file))
                
                # Determine mineral type from filename
                if 'gold' in pred_file.lower():
                    mineral = 'Gold'
                    colors = {
                        'positive': '#ffd700',
                        'negative': '#ffed4e',
                        'heatmap': ['blue', 'yellow', 'orange', 'red']
                    }
                elif 'uranium' in pred_file.lower():
                    mineral = 'Uranium'
                    colors = {
                        'positive': '#00ff00',
                        'negative': '#90ee90',
                        'heatmap': ['blue', 'green', 'yellow', 'red']
                    }
                elif 'copper' in pred_file.lower():
                    mineral = 'Copper'
                    colors = {
                        'positive': '#b87333',
                        'negative': '#cd853f',
                        'heatmap': ['blue', 'orange', 'brown', 'red']
                    }
                else:
                    mineral = 'Unknown'
                    colors = {
                        'positive': '#ff0000',
                        'negative': '#0000ff',
                        'heatmap': ['blue', 'yellow', 'red']
                    }
                
                print(f"Adding {mineral} predictions ({len(pred_gdf)} points)")
                
                # Add heatmap if probability column exists
                if 'probability' in pred_gdf.columns:
                    heat_data = [[row.geometry.y, row.geometry.x, row['probability']] 
                               for idx, row in pred_gdf.iterrows()]
                    HeatMap(
                        heat_data,
                        name=f"{mineral} Probability Heatmap",
                        radius=15,
                        gradient=colors['heatmap']
                    ).add_to(m)
                
                # Add prediction points
                if 'prediction' in pred_gdf.columns:
                    for idx, row in pred_gdf.iterrows():
                        color = colors['positive'] if row['prediction'] == 1 else colors['negative']
                        popup_text = (
                            f"<strong>{mineral}</strong><br>"
                            f"Probability: {row.get('probability', 'N/A'):.2f}<br>"
                            f"Prediction: {'Deposit' if row['prediction'] == 1 else 'No Deposit'}"
                        )
                        if 'confidence' in row:
                            popup_text += f"<br>Confidence: {row['confidence']:.2f}"
                        
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=5,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            popup=popup_text
                        ).add_to(m)
                
            except Exception as e:
                print(f"Error loading {pred_file}: {e}")
                continue
    else:
        print("No Erongo specific predictions found yet.")
        print("Check if the training script is still running or look for other prediction files.")
    
    # Check for any existing prediction files
    all_pred_files = [f for f in os.listdir(predictions_dir) if f.endswith('.shp')]
    print(f"Total prediction files found: {len(all_pred_files)}")
    
    # Show bounds of some predictions
    if all_pred_files:
        print("\nChecking bounds of first few prediction files:")
        for i, pred_file in enumerate(all_pred_files[:3]):
            try:
                pred_gdf = gpd.read_file(os.path.join(predictions_dir, pred_file))
                bounds = pred_gdf.total_bounds
                print(f"  {pred_file}: {len(pred_gdf)} points, bounds: {bounds}")
            except Exception as e:
                print(f"  Error reading {pred_file}: {e}")
    
    # Check if any predictions overlap with Erongo
    erongo_geom = erongo_gdf.geometry.iloc[0]
    overlapping_files = []
    for pred_file in all_pred_files:
        try:
            pred_gdf = gpd.read_file(os.path.join(predictions_dir, pred_file))
            # Check if any geometry in prediction file intersects with Erongo
            if pred_gdf.geometry.intersects(erongo_geom).any():
                overlapping_files.append(pred_file)
        except Exception as e:
            continue
    
    if overlapping_files:
        print(f"\nFound {len(overlapping_files)} prediction files that overlap with Erongo:")
        for pred_file in overlapping_files:
            try:
                pred_gdf = gpd.read_file(os.path.join(predictions_dir, pred_file))
                # Calculate how many points are inside Erongo
                in_erongo = pred_gdf[pred_gdf.geometry.within(erongo_geom)]
                print(f"  {pred_file}: {len(in_erongo)}/{len(pred_gdf)} points in Erongo")
            except Exception as e:
                print(f"  {pred_file}: Error checking overlap")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 180px; 
        background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
        border-radius: 10px; padding: 10px;">
            <h4 style="margin-top: 0; color: #333;">Mineral Deposit Predictions</h4>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 20px; height: 20px; background-color: #ffd700; border-radius: 50%; margin-right: 10px;"></div>
                <span>Gold Deposit</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 20px; height: 20px; background-color: #00ff00; border-radius: 50%; margin-right: 10px;"></div>
                <span>Uranium Deposit</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 20px; height: 20px; background-color: #b87333; border-radius: 50%; margin-right: 10px;"></div>
                <span>Copper Deposit</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 20px; height: 20px; background-color: #ff8c00; border: 2px solid #ff8c00; margin-right: 10px;"></div>
                <span>Erongo Boundary</span>
            </div>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save('erongo_test_map.html')
    print("\nTest map saved to: erongo_test_map.html")
    
    return m

if __name__ == "__main__":
    test_erongo_map()
