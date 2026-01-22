#!/usr/bin/env python3
"""
Extract Erongo-specific predictions from existing prediction files.
"""

import geopandas as gpd
import os

def extract_erongo_predictions():
    """Extract predictions within Erongo boundary from existing files."""
    
    print("Extracting Erongo-specific predictions...")
    
    # Load Erongo boundary
    erongo_gdf = gpd.read_file('data/erongo/Erongo.geojson')
    erongo_geom = erongo_gdf.geometry.iloc[0]
    
    # Check for existing prediction files
    predictions_dir = 'data/predictions'
    all_pred_files = [f for f in os.listdir(predictions_dir) if f.endswith('.shp')]
    
    # Find files that overlap with Erongo
    overlapping_files = []
    for pred_file in all_pred_files:
        try:
            pred_gdf = gpd.read_file(os.path.join(predictions_dir, pred_file))
            if pred_gdf.geometry.intersects(erongo_geom).any():
                overlapping_files.append(pred_file)
        except Exception as e:
            continue
    
    print(f"Found {len(overlapping_files)} overlapping prediction files")
    
    # Extract Erongo predictions from each file
    extracted_count = 0
    for pred_file in overlapping_files:
        try:
            pred_gdf = gpd.read_file(os.path.join(predictions_dir, pred_file))
            
            # Extract predictions within Erongo
            erongo_preds = pred_gdf[pred_gdf.geometry.within(erongo_geom)].copy()
            
            if len(erongo_preds) > 0:
                # Determine mineral type from existing columns or filename
                if 'mineral' in erongo_preds.columns:
                    mineral = erongo_preds['mineral'].iloc[0]
                else:
                    # Try to infer from filename
                    filename = os.path.basename(pred_file).lower()
                    if 'gold' in filename:
                        mineral = 'Gold'
                    elif 'uranium' in filename:
                        mineral = 'Uranium'
                    elif 'copper' in filename:
                        mineral = 'Copper'
                    else:
                        mineral = 'Unknown'
                
                erongo_preds['mineral'] = mineral
                
                # Save extracted predictions
                output_filename = f"erongo_{mineral.lower()}_predictions_{os.path.splitext(pred_file)[0]}.shp"
                output_path = os.path.join(predictions_dir, output_filename)
                erongo_preds.to_file(output_path, driver='ESRI Shapefile')
                
                print(f"Extracted {len(erongo_preds)} {mineral} predictions from {pred_file}")
                extracted_count += 1
                
        except Exception as e:
            print(f"Error extracting from {pred_file}: {e}")
            continue
    
    print(f"\nSuccessfully extracted {extracted_count} Erongo-specific prediction files")
    
    # Verify the extracted files
    erongo_files = [f for f in os.listdir(predictions_dir) if 'erongo' in f.lower() and f.endswith('.shp')]
    print(f"\nFound {len(erongo_files)} Erongo-specific prediction files:")
    for file in erongo_files:
        try:
            gdf = gpd.read_file(os.path.join(predictions_dir, file))
            print(f"  {file}: {len(gdf)} points")
        except Exception as e:
            print(f"  Error reading {file}: {e}")
    
    return erongo_files

def create_comprehensive_erongo_map(erongo_files):
    """Create a comprehensive map with all Erongo predictions."""
    import folium
    from folium.plugins import HeatMap
    
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
    
    # Add each mineral's predictions
    for pred_file in erongo_files:
        try:
            pred_gdf = gpd.read_file(os.path.join('data/predictions', pred_file))
            
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
            print(f"Error adding {pred_file}: {e}")
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
    m.save('comprehensive_erongo_predictions.html')
    print("\nComprehensive map saved to: comprehensive_erongo_predictions.html")
    
    return m

if __name__ == "__main__":
    erongo_files = extract_erongo_predictions()
    if erongo_files:
        create_comprehensive_erongo_map(erongo_files)
    else:
        print("No Erongo-specific predictions were extracted")
