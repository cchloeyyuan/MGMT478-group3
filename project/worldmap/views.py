from django.shortcuts import render

# This python file is used to process user requests and returning appropriate responses. 
# It determines the logic of our webpage and determines what data to display and how to update data
import folium
from .models import WeatherData

def map_view(request):
    # Get weather station data from the model
    weather_stations = WeatherData.objects.all()

    # Create a Folium map centered at the first station's location
    my_map = folium.Map(location=[weather_stations.first().LATITUDE, weather_stations.first().LONGITUDE], zoom_start=10)

    # Add markers for each weather station
    for station in weather_stations:
        popup_text = f"{station.NAME}<br>Date: {station.DATE}<br>Temperature: {station.TOBS}°C"
        folium.Marker([station.LATITUDE, station.LONGITUDE], popup=popup_text).add_to(my_map)

    # Convert the Folium map to HTML
    map_html = my_map._repr_html_()

    return render(request, 'map.html', {'map_html': map_html})

