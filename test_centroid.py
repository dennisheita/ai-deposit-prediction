import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np

# Create a test polygon
polygon = Polygon([[15.0, -22.0], [15.1, -22.0], [15.1, -22.1], [15.0, -22.1], [15.0