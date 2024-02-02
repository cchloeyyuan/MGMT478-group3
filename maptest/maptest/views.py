from django.shortcuts import render
import folium
from maptest.models import WeatherData
import pandas as pd

# Create your views here.
def map_view(request):
    # Get weather station data from the model
    weather_stations = WeatherData.objects.all()

    # If data doesn't exist in the database, insert it
    if not weather_stations:
        #file_path = "/Users/chloeyuan/Desktop/478group3/maptest/Bloomington Weather Data.csv"
        #df = pd.read_csv(file_path)
        url = 'https://raw.githubusercontent.com/cchloeyyuan/MGMT478-group3/main/Bloomington%20Weather%20Data.csv'
        df = pd.read_csv(url)


        # Fill any NaN values with True in boolean fields
        df[['WT01', 'WT03', 'WT04', 'WT05', 'WT06', 'WT11']] = df[['WT01', 'WT03', 'WT04', 'WT05', 'WT06', 'WT11']].fillna(True)

        # Convert DataFrame to a list of dictionaries and create WeatherData objects
        data_to_insert = df.to_dict(orient='records')
        WeatherData.objects.bulk_create([WeatherData(**data) for data in data_to_insert])

    # Create a Folium map centered at the first station's location
    my_map = folium.Map(location=[weather_stations.first().LATITUDE, weather_stations.first().LONGITUDE], zoom_start=10)

    # Add markers for each weather station
    for station in weather_stations:
        popup_text = f"{station.NAME}<br>Date: {station.DATE}<br>Temperature: {station.TOBS}°C"
        folium.Marker([station.LATITUDE, station.LONGITUDE], popup=popup_text).add_to(my_map)

    # Convert the Folium map to HTML
    map_html = my_map._repr_html_()

    return render(request, 'map.html', {'map_html': map_html})