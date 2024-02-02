# views.py
from django.shortcuts import render
import folium
from .models import WeatherData
import pandas as pd

def map_view(request):
    # Get weather station data from the model
    weather_stations = WeatherData.objects.all()
    #df = pd.DataFrame() #this df was initiated outside of the if statement so it can be referenced throughout the file
    file_path = "C:\\Users\\caleb\\OneDrive\\Desktop\\MGMT478-group3\\Bloomington Weather Data.csv"
    df = pd.read_csv(file_path)


    # If data doesn't exist in the database, insert it
    if not weather_stations:
        # Convert DataFrame to a list of dictionaries and create WeatherData objects
        data_to_insert = df.dropna().to_dict(orient='records')
        WeatherData.objects.bulk_create([WeatherData(**data) for data in data_to_insert])

    # Create a Folium map centered at the first station's location
    my_map = folium.Map(location=[weather_stations.first().LATITUDE, weather_stations.first().LONGITUDE], zoom_start=10)

    # Calculate average TMIN, TMAX, and TOBS for each unique station
    # Assuming your 'STATION' field is a unique identifier for each station
    station_averages = df.groupby('STATION').agg({
        'NAME': 'first',
        'LATITUDE': 'first', 
        'LONGITUDE': 'first',
        'TMIN': 'mean',
        'TMAX': 'mean',
        'TOBS': 'mean'
    }).reset_index()

    # Add marker for each unique weather station with average values
    for index, row in station_averages.iterrows():
        popup_text = f"{row['STATION']}<br>Name: {row['NAME']}<br>Avg TMIN: {row['TMIN']}°C<br>Avg TMAX: {row['TMAX']}°C<br>Avg TOBS: {row['TOBS']}°C"
        folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=popup_text).add_to(my_map)

    # Convert the Folium map to HTML
    map_html = my_map._repr_html_()

    return render(request, 'map.html', {'map_html': map_html})
