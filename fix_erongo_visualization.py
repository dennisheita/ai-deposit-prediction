#!/usr/bin/env python3
"""
Fixed Erongo visualization.
"""

import geopandas as gpd
import folium
from folium.plugins import HeatMap
import os

def create_comprehensive_erongo_map():
    """Create a comprehensive map with all Erongo predictions."""
    
    print("Creating comprehensive Erongo map...")
    
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
    
    # Color schemes per mineral
    color_schemes = {
        'Gold': {
            'positive': '#ffd700',
            'negative': '#ffed4e',
            'heatmap': ['blue', 'yellow', 'orange', 'red']
        },
        'Uranium': {
            'positive': '#00ff00',
            'negative': '#90ee90',
            'heatmap': ['blue', 'green', 'yellow', 'red']
        },
        'Copper': {
            'positive': '#b87333',
            'negative': '#cd853f',
            'heatmap': ['blue', 'orange', 'brown', 'red']
        },
        'Unknown': {
            'positive': '#ff0000',
            'negative': '#0000ff',
            'heatmap': ['blue', 'yellow', 'red']
        }
    }
    
    # Find Erongo-specific prediction files
    predictions_dir = 'data/predictions'
    erongo_files = [f for f in os.listdir(predictions_dir) if 'erongo' in f.lower() and f.endswith('.shp')]
    
    print(f"Found {len(erongo_files)} Erongo-specific prediction files")
    
    # Add each mineral's predictions
    for pred_file in erongo_files:
        try:
            pred_gdf = gpd.read_file(os.path.join('data/predictions', pred_file))
            
            # Skip small files (likely just single points)
            if len(pred_gdf) < 100:
                print(f"Skipping small file: {pred_file} ({len(pred_gdf)} points)")
                continue
            
            # Determine mineral type
            if 'mineral' in pred_gdf.columns:
                mineral = pred_gdf['mineral'].iloc[0]
            else:
                filename = os.path.basename(pred_file).lower()
                if 'gold' in filename:
                    mineral = 'Gold'
                elif 'uranium' in filename:
                    mineral = 'Uranium'
                elif 'copper' in filename:
                    mineral = 'Copper'
                else:
                    mineral = 'Unknown'
            
            colors = color_schemes.get(mineral, color_schemes['Unknown'])
            
            print(f"Adding {mineral} predictions ({len(pred_gdf)} points)")
            
            # Check for probability column (shapefiles truncate to 10 chars)
            prob_col = 'probabilit' if 'probabilit' in pred_gdf.columns else 'probability'
            
            # Add heatmap if probability column exists
            if prob_col in pred_gdf.columns:
                heat_data = [[row.geometry.y, row.geometry.x, row[prob_col]] 
                           for idx, row in pred_gdf.iterrows()]
                HeatMap(
                    heat_data,
                    name=f"{mineral} Probability Heatmap",
                    radius=15,
                    gradient=colors['heatmap']
                ).add_to(m)
            
            # Check for prediction column
            pred_col = 'predictio' if 'predictio' in pred_gdf.columns else 'prediction'
            
            if pred_col in pred_gdf.columns:
                # Add prediction points - only show positive predictions to reduce clutter
                positive_preds = pred_gdf[pred_gdf[pred_col] == 1]
                if len(positive_preds) > 0:
                    for idx, row in positive_preds.iterrows():
                        popup_text = f"<strong>{mineral}</strong><br>"
                        
                        if prob_col in pred_gdf.columns:
                            prob_val = row[prob_col]
                            popup_text += f"Probability: {prob_val:.2f}<br>"
                        
                        popup_text += f"Prediction: Deposit"
                        
                        # Check for confidence column
                        conf_col = 'confidenc' if 'confidenc' in pred_gdf.columns else 'confidence'
                        if conf_col in pred_gdf.columns:
                            conf_val = row[conf_col]
                            popup_text += f"<br>Confidence: {conf_val:.2f}"
                        
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=3,
                            color=colors['positive'],
                            fill=True,
                            fill_color=colors['positive'],
                            fill_opacity=0.6,
                            popup=popup_text
                        ).add_to(m)
            
        except Exception as e:
            print(f"Error adding {pred_file}: {e}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            continue
    
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
    m.save('comprehensive_erongo_predictions_fixed.html')
    print("\nFixed map saved to: comprehensive_erongo_predictions_fixed.html")
    
    return m

if __name__ == "__main__":
    create_comprehensive_erongo_map()
