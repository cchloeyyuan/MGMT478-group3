# views.py
from django.shortcuts import render
import folium
from folium.plugins import HeatMap
from .models import WeatherData
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from django.shortcuts import render
from .forms import CoordinatesForm
from django.http import HttpResponseRedirect
from django.urls import reverse
import requests, os
import numpy as np
from scipy.interpolate import griddata

def map_view(request):
    # Get weather station data from the model
    weather_stations = WeatherData.objects.all()
    url = 'https://raw.githubusercontent.com/cchloeyyuan/MGMT478-group3/main/Indiana%20Weather%20Data.csv'
    df = pd.read_csv(url)

    # If data doesn't exist in the database, insert it
    if not weather_stations:
        # Convert DataFrame to a list of dictionaries and create WeatherData objects
        data_to_insert = df.dropna().to_dict(orient='records')
        WeatherData.objects.bulk_create([WeatherData(**data) for data in data_to_insert])

    # Create a Folium map centered at the first station's location
    my_map = folium.Map(location=[weather_stations.first().LATITUDE, weather_stations.first().LONGITUDE], zoom_start=10)
    
    # Calculate average weather statistics for each station
    # Assuming your 'STATION' field is a unique identifier for each station
    numeric_cols = ['AWND', 'PRCP', 'SNOW', 'TAVG', 'TMIN', 'TMAX']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    station_averages = df.groupby('STATION').agg({
        'NAME': 'first',
        'LATITUDE': 'first', 
        'LONGITUDE': 'first',
        'AWND': 'mean',
        'PRCP': 'mean',
        'SNOW': 'mean',
        'TAVG': 'mean',
        'TMIN': 'mean',
        'TMAX': 'mean'
    }).reset_index()


    # Give boundaries for grid
    min_lat, max_lat = 40.15, 40.8
    min_lon, max_lon = -87.7, -86

    # Create a grid of points covering Indiana region
    grid_lat, grid_lon = np.mgrid[min_lat:max_lat:0.08, min_lon:max_lon:0.08]
        
    # Interpolate PRCP values for each point in the grid
    interpolated_prcp = griddata((station_averages['LATITUDE'], station_averages['LONGITUDE']),
                                station_averages['PRCP'],
                                (grid_lat, grid_lon),
                                method='cubic')

    # Replace NaN values with a default value (e.g., 0)
    interpolated_prcp = np.nan_to_num(interpolated_prcp)
    heatmap_data = []

    # Iterate over the grid coordinates and corresponding interpolated precipitation values
    for lat, lon, prcp in zip(grid_lat.flatten(), grid_lon.flatten(), interpolated_prcp.flatten()):
        heatmap_data.append((lat, lon, prcp))

    #Add in original weather station data to the heatmap data
    for index, row in station_averages.iterrows():
        lat = row['LATITUDE']
        lon = row['LONGITUDE']
        prcp = row['PRCP']
        heatmap_data.append((lat, lon, prcp))

    # Determine the maximum interpolated precipitation values in order to normalize values
    max_value = max(entry[2] for entry in heatmap_data)
    normalized_heatmap_data = []
    for entry in heatmap_data: #this loop goes through all the precipitation values and normalizes them from 0-1
        lat, lon, prcp = entry  # Unpack the tuple
        normalized_prcp = prcp / max_value
        normalized_heatmap_data.append((lat, lon, normalized_prcp)) 

    # Add heatmap overlay
    HeatMap(normalized_heatmap_data, gradient={0.3:'blue',
    0.5: 'green',
    0.75: 'yellow',
    0.9: 'orange',
    1.0: 'red'}, index=2).add_to(my_map)
    

    # Add marker for each unique weather station with average values
    for index, row in station_averages.iterrows():
        popup_text = f"{row['STATION']}<br>Name: {row['NAME']}<br>Avg TMIN: {row['TMIN']}°C<br>Avg TMAX: {row['TMAX']}°C"
        folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=popup_text).add_to(my_map)


    countey_data = gpd.read_file("counties.geojson")
    #add all USA counties to the map
    folium.GeoJson("counties.geojson").add_to(my_map)
    # convert weather data into geodataframe
 #   geometry = [Point(xy) for xy in df]
 #   weatherstations_gdf = gpd.GeoDataFrame(df['longitude'], df['latitude'], geometry = geometry, crs = countey_data.crs)
    
    # perform spatial join to find which countey each weather station falls within
 #   joined_data = gpd.sjoin(countey_data, weatherstations_gdf, how = "inner", op = "contains")

    # Convert the Folium map to HTML
    map_html = my_map._repr_html_()
 #   print(joined_data)

    return render(request, 'map.html', {'map_html': map_html})

def map_request(request):
    # Initialize the map with some default state
    #my_map = folium.Map(location=[default_latitude, default_longitude], zoom_start=10)
    form = CoordinatesForm()

    if request.method == 'POST':
        form = CoordinatesForm(request.POST)
        if form.is_valid():
            latitude = form.cleaned_data['latitude']
            longitude = form.cleaned_data['longitude']
            lat_direction = form.cleaned_data['lat_direction']
            long_direction = form.cleaned_data['long_direction']

            # Adjust latitude and longitude based on direction
            latitude = latitude if lat_direction == 'N' else -latitude
            longitude = longitude if long_direction == 'E' else -longitude
            

            my_map = folium.Map(location=[float(latitude), float(longitude)], zoom_start=14)
            folium.Marker([float(latitude), float(longitude)], popup="Your Location").add_to(my_map)
            map_html = my_map._repr_html_()
            
            return render(request, 'map.html', {'map_html': map_html})
            
        else:
            # 如果是GET请求，仅渲染带有空表单的页面
            # If it's a GET request, only render the page with the empty form
            form = CoordinatesForm()
            return render(request, 'map.html', {'form': form})# Create a new map object with the submitted coordinates
            #my_map = folium.Map(location=[latitude, longitude], zoom_start=14)
            #folium.Marker([latitude, longitude], popup="Your Location").add_to(my_map)

    #map_html = my_map._repr_html_()
    #return render(request, 'map.html', {'form': form, 'map_html': map_html})