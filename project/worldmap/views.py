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
from sklearn.neighbors import NearestNeighbors
from django.http import JsonResponse
from django.core.mail import send_mail
from django.conf import settings
from django.core.mail import send_mail
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from shapely.geometry import Polygon

def map_view(request):
    # Get weather station data from the model
    weather_stations = WeatherData.objects.all()
    url = 'https://raw.githubusercontent.com/cchloeyyuan/MGMT478-group3/main/Indiana%20Weather%20Data.csv'
    file_path = r"C:\Users\caleb\OneDrive\Desktop\Indiana Weather Data.csv"
    df = pd.read_csv(file_path)

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

    desired_state_code = "18"
    desired_color_value = "PRCP"
    coordinates = []
    heatmap_data = []

    with open("counties.geojson", "r") as file:
        data = json.load(file)
        for feature in data["features"]:
            properties = feature["properties"]
            geometry = feature["geometry"]
            if properties.get("STATEFP") == desired_state_code:
                coordinates.extend(geometry["coordinates"][0]) 
                heatmap_data.append(heatmap(station_averages, np.array(coordinates), desired_color_value))

    weighted_values = []

    for entry in heatmap_data:
        for inner_entry in entry:
            weighted_value= inner_entry[2] 
            weighted_values.append(weighted_value)

    max_value = max(weighted_values)
    avg_value = np.mean(weighted_values)

    normalized_heatmap_data = []

    for entry in heatmap_data:
        for inner_entry in entry:
            lat = inner_entry[0]
            lon = inner_entry[1]
            weighted_value= inner_entry[2] 

        if not np.isnan(weighted_value):  # Check if the weighted value is not NaN
            normalized_value = (weighted_value - avg_value) / max_value if max_value else 0  # Normalize the value
            normalized_heatmap_data.append((lat, lon, normalized_value))

    # Add heatmap overlay
    HeatMap(normalized_heatmap_data, gradient={0.3: 'blue', 0.5: 'green', 0.75: 'yellow', 0.9: 'orange', 1.0: 'red'}, index=2).add_to(my_map)

        

    # Add marker for each unique weather station with average values
    for index, row in station_averages.iterrows():
        popup_text = f"{row['STATION']}<br>Name: {row['NAME']}<br>Avg TMIN: {row['TMIN']}°C<br>Avg TMAX: {row['TMAX']}°C"
        folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=popup_text).add_to(my_map)


    countey_data = gpd.read_file("counties.geojson")
    #add all USA counties to the map
    #folium.GeoJson("counties.geojson").add_to(my_map)
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


@csrf_exempt
def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        sender_email = request.POST.get('email')  # 发件人邮箱，用户输入的
        subject = request.POST.get('subject')
        message = request.POST.get('message')
        recipient_email = request.POST.get('recipient')  # 接收方的邮箱，也是用户输入的

        try:
            send_mail(
                subject,
                f"Name: {name}\nEmail: {sender_email}\nMessage: {message}",
                sender_email,  # 这将作为回复地址
                [recipient_email],  # 发送到用户输入的这个地址
                fail_silently=False,
            )
            return HttpResponse('Your message has been sent. Thank you!')
        except Exception as e:
            return HttpResponse(f'An error occurred: {e}')

    return HttpResponse('This endpoint only accepts POST requests.')

    #map_html = my_map._repr_html_()
    #return render(request, 'map.html', {'form': form, 'map_html': map_html})

def heatmap(station_averages, county_coords, color_value):

    grid_lat = county_coords[:, 1]
    grid_lon = county_coords[:, 0]

    # Fit KNN model on station coordinates
    knn_model = NearestNeighbors(n_neighbors=3).fit(station_averages[['LATITUDE', 'LONGITUDE']])

    # Find indices of the 3 closest stations for each grid point
    _, indices = knn_model.kneighbors(np.column_stack((grid_lat, grid_lon)))
    weighted_values = []

    for index_set in indices:
        closest_station_value = station_averages.iloc[index_set[0]][color_value]
        second_closest_station_value = station_averages.iloc[index_set[1]][color_value]
        third_closest_station_value = station_averages.iloc[index_set[2]][color_value]

        # Assign weights to the values from closest, second closest, and third closest stations
        weighted_value = (closest_station_value * 0.75 +
                          second_closest_station_value * 0.2 +
                          third_closest_station_value * 0.05)

        weighted_values.append(weighted_value)

    # Assign the same weighted value to all coordinates within each county
    return [(grid_lat[i], grid_lon[i], weighted_values[i]) for i in range(len(grid_lat))]