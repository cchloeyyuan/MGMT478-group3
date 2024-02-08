# views.py
from django.shortcuts import render
import folium
from folium.plugins import HeatMap
import requests
from .models import WeatherData
import pandas as pd

def map_view(request):
    # Get weather station data from the model
    weather_stations = WeatherData.objects.all()
    #df = pd.DataFrame() #this df was initiated outside of the if statement so it can be referenced throughout the file
    #file_path = "C:\\Users\\caleb\\OneDrive\\Desktop\\MGMT478-group3\\Bloomington Weather Data.csv"
    #df = pd.read_csv(file_path)
    url = 'https://raw.githubusercontent.com/cchloeyyuan/MGMT478-group3/main/Bloomington%20Weather%20Data.csv'
    df = pd.read_csv(url)

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

    # add the heatmap test to see if run
    heat_map_data = station_averages[['LATITUDE', 'LONGITUDE']].values.tolist()
    HeatMap(heat_map_data).add_to(my_map)

    # Add marker for each unique weather station with average values
    for index, row in station_averages.iterrows():
        popup_text = f"{row['STATION']}<br>Name: {row['NAME']}<br>Avg TMIN: {row['TMIN']}°C<br>Avg TMAX: {row['TMAX']}°C<br>Avg TOBS: {row['TOBS']}°C"
        folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=popup_text).add_to(my_map)


    #Get link to JSON file folders
    #geoJSON_Files = 'https://github.com/cchloeyyuan/MGMT478-group3/tree/7b2b9305b13fdf4ac1bfb8fe1a8b2281e50eb5fd/State-zip-code-GeoJSON-master'
    #response = requests.get(geoJSON_Files)
    

    # Check if the request was successful
    #if response.status_code == 200:
        # Parse the JSON response
    #    folder_contents = response.json()
    #    print(folder_contents)

        # Count the number of files in the folder
    #    file_count = len(folder_contents)
         #iterate through all files in folders
    #    for filename in geoJSON_Files:
    #        if '.json' in folder_contents.values():
                #create url for json file
    #            cordfile =  folder_contents['path']
    #            cordfile.remove('State-zip-code-GeoJSON-master')
    #            cordfile = geoJSON_Files + cordfile 
    #            HeatMap(cordfile).add_to(my_map)
                # print(os.path.join(directory, filename))
    #        else:
    #              continue
    
    
    # Convert the Folium map to HTML
    map_html = my_map._repr_html_()

    return render(request, 'map.html', {'map_html': map_html})
