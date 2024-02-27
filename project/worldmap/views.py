# views.py
from django.shortcuts import render
import folium
from folium.plugins import HeatMap
from .models import WeatherData
import pandas as pd
from django.shortcuts import render
from .forms import CoordinatesForm
from django.http import HttpResponseRedirect
from django.urls import reverse
import requests, os

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

    # add the heatmap test to see if run
    heat_map_data = station_averages[['LATITUDE', 'LONGITUDE']].values.tolist()
    HeatMap(heat_map_data).add_to(my_map)
    

    
    # Add marker for each unique weather station with average values
    for index, row in station_averages.iterrows():
        popup_text = f"{row['STATION']}<br>Name: {row['NAME']}<br>Avg TMIN: {row['TMIN']}°C<br>Avg TMAX: {row['TMAX']}°C"
        folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=popup_text).add_to(my_map)

    # get local filepath for project
    current_loc = os.getcwd()
    # add zip code folder to filepath
#    final_directory = os.path.join(current_loc, r'State-zip-code-GeoJSON-master/')
#    i = 0
    # loop through file for all files
#    for file in os.listdir(final_directory):
        #break after 3 json files are added
#        if i == 4:
#            break
        # if json file
#        if file.endswith(".json"):
#            i+=1
            # get filepath of json file
#            json_file = os.path.join(final_directory, file)
            #add json file to map
#            folium.GeoJson(json_file).add_to(my_map)  
    folium.GeoJson("counties.geojson").add_to(my_map)

    # Convert the Folium map to HTML
    map_html = my_map._repr_html_()

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