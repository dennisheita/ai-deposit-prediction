#!/usr/bin/env python3
"""
Create a quick Erongo map with a small sample of predictions.
"""

import geopandas as gpd
import folium
from folium.plugins import HeatMap

def create_quick_erongo_map():
    """Create a quick map with a small sample of predictions."""
    
    print("Creating quick Erongo map...")
    
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
    
    # Find the largest Erongo prediction file
    import os
    predictions_dir = 'data/predictions'
    erongo_files = [f for f in os.listdir(predictions_dir) if 'erongo' in f.lower() and f.endswith('.shp')]
    
    largest_file = None
    max_size = 0
    
    for pred_file in erongo_files:
        try:
            gdf = gpd.read_file(os.path.join(predictions_dir, pred_file))
            if len(gdf) > max_size and len(gdf) > 100:
                max_size = len(gdf)
                largest_file = pred_file
        except Exception as e:
            continue
    
    if largest_file:
        print(f"Using file: {largest_file} (sampling {min(1000, max_size)} points)")
        
        try:
            pred_gdf = gpd.read_file(os.path.join(predictions_dir, largest_file))
            
            # Take a random sample for quick visualization
            sample_size = min(1000, len(pred_gdf))
            sampled_gdf = pred_gdf.sample(n=sample_size, random_state=42)
            
            # Check for probability column (shapefiles truncate to 10 chars)
            prob_col = 'probabilit' if 'probabilit' in sampled_gdf.columns else 'probability'
            
            # Add heatmap if probability column exists
            if prob_col in sampled_gdf.columns:
                heat_data = [[row.geometry.y, row.geometry.x, row[prob_col]] 
                           for idx, row in sampled_gdf.iterrows()]
                HeatMap(
                    heat_data,
                    name="Probability Heatmap",
                    radius=15,
                    gradient=['blue', 'yellow', 'orange', 'red']
                ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add legend
            legend_html = '''
                <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                border-radius: 10px; padding: 10px;">
                    <h4 style="margin-top: 0; color: #333;">Mineral Deposit Predictions</h4>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 20px; height: 20px; background-color: #ff8c00; border: 2px solid #ff8c00; margin-right: 10px;"></div>
                        <span>Erongo Boundary</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background: linear-gradient(to right, blue, yellow, red); border-radius: 50%; margin-right: 10px;"></div>
                        <span>Probability Heatmap</span>
                    </div>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Save map
            m.save('quick_erongo_map.html')
            print("Quick map saved to: quick_erongo_map.html")
            
        except Exception as e:
            print(f"Error adding predictions: {e}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
    else:
        print("No suitable Erongo prediction files found")
    
    return m

if __name__ == "__main__":
    create_quick_erongo_map()
