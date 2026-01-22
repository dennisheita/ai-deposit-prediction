#!/usr/bin/env python3
"""
Visualize Erongo predictions on an interactive map.
"""

import geopandas as gpd
import folium
from folium.plugins import HeatMap
import os

# Directories
DATA_DIR = 'data'
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
ERONGO_DIR = os.path.join(DATA_DIR, 'erongo')


def load_erongo_boundary():
    """Load Erongo boundary from GeoJSON."""
    erongo_path = os.path.join(ERONGO_DIR, 'Erongo.geojson')
    erongo_gdf = gpd.read_file(erongo_path)
    erongo_gdf = erongo_gdf.to_crs('EPSG:4326')  # Ensure WGS84
    return erongo_gdf


def load_predictions():
    """Load prediction shapefiles."""
    predictions = []
    for file in os.listdir(PREDICTIONS_DIR):
        if file.endswith('.shp') and 'erongo' in file.lower():
            try:
                file_path = os.path.join(PREDICTIONS_DIR, file)
                gdf = gpd.read_file(file_path)
                gdf['filename'] = file
                # Extract mineral type
                if 'gold' in file.lower():
                    gdf['mineral'] = 'Gold'
                elif 'uranium' in file.lower():
                    gdf['mineral'] = 'Uranium'
                elif 'copper' in file.lower():
                    gdf['mineral'] = 'Copper'
                predictions.append(gdf)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return predictions


def create_erongo_map(predictions, erongo_gdf):
    """Create interactive map of Erongo predictions."""
    # Calculate map center
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

    # Add predictions for each mineral
    for gdf in predictions:
        mineral = gdf['mineral'].iloc[0]
        
        # Define color scheme based on mineral
        color_scheme = {
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
            }
        }
        
        colors = color_scheme.get(mineral, color_scheme['Gold'])
        
        # Add heatmap layer
        if 'probability' in gdf.columns:
            heat_data = [[row.geometry.y, row.geometry.x, row['probability']] 
                       for idx, row in gdf.iterrows()]
            HeatMap(
                heat_data,
                name=f"{mineral} Probability Heatmap",
                radius=15,
                gradient=colors['heatmap']
            ).add_to(m)
        
        # Add prediction points
        if 'prediction' in gdf.columns:
            # Positive predictions (deposits)
            positive_gdf = gdf[gdf['prediction'] == 1]
            for idx, row in positive_gdf.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color=colors['positive'],
                    fill=True,
                    fill_color=colors['positive'],
                    fill_opacity=0.7,
                    popup=(
                        f"<strong>{mineral}</strong><br>"
                        f"Probability: {row.get('probability', 'N/A'):.2f}<br>"
                        f"Prediction: Deposit<br>"
                        f"Confidence: {row.get('confidence', 'N/A'):.2f}"
                    )
                ).add_to(m)
            
            # Negative predictions (no deposits)
            negative_gdf = gdf[gdf['prediction'] == 0]
            for idx, row in negative_gdf.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    color=colors['negative'],
                    fill=True,
                    fill_color=colors['negative'],
                    fill_opacity=0.3,
                    popup=(
                        f"<strong>{mineral}</strong><br>"
                        f"Probability: {row.get('probability', 'N/A'):.2f}<br>"
                        f"Prediction: No Deposit<br>"
                        f"Confidence: {row.get('confidence', 'N/A'):.2f}"
                    )
                ).add_to(m)

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

    return m


def create_mineral_specific_map(predictions, mineral, erongo_gdf):
    """Create mineral-specific map."""
    mineral_gdf = [gdf for gdf in predictions if gdf['mineral'].iloc[0] == mineral][0]
    
    center_lat = mineral_gdf.geometry.centroid.y.mean()
    center_lon = mineral_gdf.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

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

    # Add heatmap
    if 'probability' in mineral_gdf.columns:
        heat_data = [[row.geometry.y, row.geometry.x, row['probability']] 
                   for idx, row in mineral_gdf.iterrows()]
        HeatMap(
            heat_data,
            name=f"{mineral} Probability Heatmap",
            radius=15
        ).add_to(m)

    # Add prediction points
    if 'prediction' in mineral_gdf.columns:
        for idx, row in mineral_gdf.iterrows():
            color = 'red' if row['prediction'] == 1 else 'blue'
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

    folium.LayerControl().add_to(m)

    return m


def main():
    """Main function."""
    # Load data
    erongo_gdf = load_erongo_boundary()
    predictions = load_predictions()
    
    if not predictions:
        print("No prediction files found. Please run train_predict_erongo.py first.")
        return
    
    print(f"Loaded {len(predictions)} prediction datasets")
    
    # Create combined map
    combined_map = create_erongo_map(predictions, erongo_gdf)
    combined_map.save('erongo_predictions_map.html')
    print("Combined map saved to: erongo_predictions_map.html")
    
    # Create mineral-specific maps
    for mineral in ['Gold', 'Uranium', 'Copper']:
        mineral_predictions = [gdf for gdf in predictions if gdf['mineral'].iloc[0] == mineral]
        if mineral_predictions:
            mineral_map = create_mineral_specific_map(mineral_predictions, mineral, erongo_gdf)
            mineral_map.save(f'erongo_{mineral.lower()}_predictions.html')
            print(f"{mineral} map saved to: erongo_{mineral.lower()}_predictions.html")


if __name__ == "__main__":
    main()
