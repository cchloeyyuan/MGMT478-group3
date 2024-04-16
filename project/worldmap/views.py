# views.py
from django.shortcuts import render
import folium
from folium.plugins import HeatMap
from .models import GlobalData
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
import warnings
from .forms import TimePeriodForm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning


def map_view(request):
    # Get weather station data from the model
    weather_stations = GlobalData.objects.all()
    # Convert the QuerySet to a Pandas DataFrame
    df = pd.DataFrame(list(weather_stations.values()))
 
    # Check if the DataFrame is empty
    if not df.empty:
        # Create a Folium map centered at the first station's location
        my_map = folium.Map(location=[df.iloc[0]['Latitude'], df.iloc[0]['Longitude']], zoom_start=10)
        
    # Create a Folium map centered at the first station's location
    my_map = folium.Map(location=[weather_stations.first().Latitude, weather_stations.first().Longitude], zoom_start=10)
    
    # Calculate average weather statistics for each station
    # Assuming your 'STATION' field is a unique identifier for each station
    numeric_cols = ['AWND', 'PRCP', 'TAVG', 'TMIN', 'TMAX']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    station_averages = df.groupby('station_id').agg({
        'station_id': 'first',
        'Latitude': 'first',
        'Longitude': 'first',
        'AWND': 'mean',
        'PRCP': 'mean',
        'TAVG': 'mean',
        'TMIN': 'mean',
        'TMAX': 'mean'
    }).reset_index(drop=True)

    desired_state_code = "18" #This is the state code value that specifies which counties to perform KNN on
    desired_color_value = "PRCP" # This is the column name that is used for the values of the heatmap
    heatmap_df = gpd.read_file("counties.geojson")
    heatmap_df['Measure'] = 0 #This creates a new column to store values for the heatmap

    # Loop through each county and apply the heatmap knn function to give it a measure value
    for index,row in heatmap_df.iterrows():
        if row['STATEFP'] == desired_state_code:
            measure = heatmap(station_averages, row['geometry'], desired_color_value)
            heatmap_df.loc[index, 'Measure'] = measure
        else:
            heatmap_df.loc[index, 'Measure'] = 0

    # Add marker for each unique weather station with average values
    for index, row in station_averages.iterrows():
        popup_text = f"{row['station_id']}<br>Avg TMIN: {row['TMIN']}°C<br>Avg TMAX: {row['TMAX']}°C"
        folium.Marker([row['Latitude'], row['Longitude']], popup=popup_text).add_to(my_map)

    # # Create Folium map
        # Add choropleth layer to the map
    folium.Choropleth(
        geo_data=heatmap_df,
        name='choropleth',
        data=heatmap_df,
        columns=['GEOID', 'Measure'],
        key_on='feature.properties.GEOID',
        fill_color='YlOrRd',  # Change the color scale if needed
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Average Precipitation (mm)'
    ).add_to(my_map)

    # Add layer control
    folium.LayerControl().add_to(my_map)
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


@csrf_exempt
def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        sender_email = request.POST.get('email')  # user input
        subject = request.POST.get('subject')
        message = request.POST.get('message')
        recipient_email = request.POST.get('recipient')  # 
        try:
            send_mail(
                f"Reply to {subject}",
                f"Dear {name},\n\n"
                f"Thanks for your message.\n\n"
                f"This email means we have received your message and will get back to you as soon as possible.\n\n"
                f"Appreciate your time! \n\n\n "
                f"------------------------Below is the orginal message we received -----------------------------\n\n"
                f"Name: {name}\n\nEmail: {sender_email}\n\nMessage: {message}",
                sender_email,  # reply adress
                [recipient_email],  # target email adress
                fail_silently=False,
                
            )
            return JsonResponse({'success': True})
        except Exception as e:
            # if fail use JsonResponse tu return message
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'This endpoint only accepts POST requests.'})

    #map_html = my_map._repr_html_()
    #return render(request, 'map.html', {'form': form, 'map_html': map_html})

def heatmap(station_averages, county_coords, color_value):

    weighted_values = []
    coordinates = county_coords.exterior.coords

    # Extract latitude and longitude coordinates
    latitudes = [coord[1] for coord in coordinates]
    longitudes = [coord[0] for coord in coordinates]

    # Fit KNN model on station coordinates
    warnings.filterwarnings("ignore", category=UserWarning)
    knn_model = NearestNeighbors(n_neighbors=3).fit(station_averages[['Latitude', 'Longitude']])

    # Find indices of the 3 closest stations for each grid point
    _, indices = knn_model.kneighbors(np.column_stack((latitudes, longitudes)))
    for index_set in indices:
        # Get the distances to the closest stations
        distances = knn_model.kneighbors()[0]
        # Calculate the weights based on the inverse of distance
        weights = 1 / distances
        weights = weights/sum(weights)

        # Get the values from closest, second closest, and third closest stations
        closest_station_value = station_averages.iloc[index_set[0]][color_value]
        second_closest_station_value = station_averages.iloc[index_set[1]][color_value]
        third_closest_station_value = station_averages.iloc[index_set[2]][color_value]

        # Calculate the weighted value using the calculated weights
        weighted_value = (closest_station_value * weights[0] +
                          second_closest_station_value * weights[1] +
                          third_closest_station_value * weights[2])

        weighted_values.append(weighted_value)

    # Calculate the mean of weighted values using NumPy
    mean_weighted_value = np.mean(weighted_values)

    return mean_weighted_value

def time_period_request(request):
    # If it's a GET request or the form is not valid, render the page with the form and any data
    form = TimePeriodForm(request.POST)  # Initialize form with POST data or None
    data_for_period = None
    
    
    if request.method == 'POST':
        # Initialize the form with POST data
        form = TimePeriodForm(request.POST)
        if form.is_valid():
            # Extract the time period information from the form
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']

            # Query your data model to find data within the time period
            data_for_period = weather_stations.objects.filter(
                DATE__gte=start_date, 
                DATE__lte=end_date
            )
        # In case the form is not valid, 'form' context will carry the errors
        # If it's valid, 'data_for_period' will carry the filtered data

    # Render the same 'map.html' for both GET and POST requests
    # This will ensure that the form persists with the user's input or with errors
    return render(request, 'map.html', {
        'form': form,
        'data': data_for_period
    })

def lasso_prediction(data):
    data['DATE'] = pd.to_datetime(data['DATE'])
    # Extract year and month from DATE as new features
    data['YEAR'] = data['DATE'].dt.year
    data['MONTH'] = data['DATE'].dt.month
    # Drop the 'SNOW' column
    data_cleaned = data.drop(['SNOW'], axis=1)
    # Convert non-numeric to numeric
    for column in ['LATITUDE','LONGITUDE','ELEVATION','AWND', 'TAVG', 'TMAX', 'TMIN']:
        data_cleaned[column] = pd.to_numeric(data_cleaned[column], errors='coerce')
    # Imputer missing data as median of the column
    imputer = SimpleImputer(strategy='median')
    data_cleaned[['AWND', 'TAVG', 'TMAX', 'TMIN']] = imputer.fit_transform(data_cleaned[['AWND', 'TAVG', 'TMAX', 'TMIN']])
    stations = data_cleaned[['STATION', 'NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
    coordinates = stations[['LATITUDE', 'LONGITUDE']]
    neighbors_model = NearestNeighbors(n_neighbors=6)
    neighbors_model.fit(coordinates)

    #Get six nearest weather stations around a given specific weather station
    def six_nearest_weather_stations(latitude, longitude):
        query_coordinates = np.array([[latitude, longitude]])
        distances, indices = neighbors_model.kneighbors(query_coordinates)
        nearest_stations_info = stations.iloc[indices[0]].copy() 
        nearest_stations_info['DISTANCE(°)'] = distances[0]

        return nearest_stations_info
    
    #Get the average value for the nearest weather station, exclude the one used as predicted weather station
    def average_values_for_nearest_stations_exclude(latitude, longitude, station_to_exclude):
        nearest_stations_info = six_nearest_weather_stations(latitude, longitude)
        nearest_station_ids = nearest_stations_info['STATION'].tolist()
        
        # Remove the specific station ID from the list
        if station_to_exclude in nearest_station_ids:
            nearest_station_ids.remove(station_to_exclude)
        
        filtered_data = data_cleaned[data_cleaned['STATION'].isin(nearest_station_ids)]
        average_values = filtered_data.groupby(['YEAR', 'MONTH'])[['AWND', 'PRCP', 'TAVG', 'TMAX', 'TMIN']].mean().reset_index()
        average_values.rename(columns={
            'AWND': 'AWND_avg',
            'PRCP': 'PRCP_avg',
            'TAVG': 'TAVG_avg',
            'TMAX': 'TMAX_avg',
            'TMIN': 'TMIN_avg'
        }, inplace=True)
        return average_values
    
    #Expand the dataset
    def get_analysis_data(latitude, longitude, weather_station):
        weather_station_data = data_cleaned[data_cleaned['STATION']== weather_station]
        
        for var in ['AWND', 'PRCP', 'TAVG', 'TMAX', 'TMIN']:
            for year in range(1, 6):
                year_lag = year*12
                weather_station_data[f'{var}_lag_{year}_year'] = weather_station_data[var].shift(year_lag)
                
        neighbor_data = average_values_for_nearest_stations_exclude(latitude, longitude, weather_station)
        merged_data = pd.merge(weather_station_data, neighbor_data, on=['YEAR', 'MONTH'], how='inner')
        
        for var in ['AWND_avg', 'PRCP_avg', 'TAVG_avg', 'TMAX_avg', 'TMIN_avg']:
            for year in range(1, 6):
                year_lag = year*12
                merged_data[f'{var}_lag_{year}_year'] = merged_data[var].shift(year_lag)
        
        merged_data_final = merged_data.drop(columns=['AWND', 'TAVG', 'TMAX', 'TMIN', 'AWND_avg', 'PRCP_avg', 'TAVG_avg', 'TMAX_avg', 'TMIN_avg'])
        merged_data_final = merged_data_final.dropna()
        return merged_data_final
    
    #Lasso - Top 5 Feature Selection
    def lasso_mse(latitude, longitude, weather_station):
        
        merged_data_final = get_analysis_data(latitude, longitude, weather_station)
        
        # Initialize dictionaries to store MSE values for each year and top features for each year
        mse_values = {}
        top_features_per_year = {}

        # Initialize lists to accumulate actual and predicted values for all years
        all_actuals = []
        all_predictions = []

        # Set the starting and ending years for the time window
        years = merged_data_final['YEAR'].unique()
        start_year = years[0] + 4
        end_year = 2022

        # Loop through each time window
        for year in range(start_year, end_year + 1):
            # Define the training and testing sets
            train_df = merged_data_final[merged_data_final['YEAR'].between(year - 4, year)]
            test_df = merged_data_final[merged_data_final['YEAR'] == year + 1]

            # Remove rows with missing values
            train_df = train_df.dropna()
            test_df = test_df.dropna()

            # Select features and target variable
            X_train = train_df.drop(columns=['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'DATE', 'PRCP', 'YEAR'])
            y_train = train_df['PRCP']
            X_test = test_df[X_train.columns]
            y_test = test_df['PRCP']

            # Use LassoCV for feature selection and model fitting
            lasso = LassoCV(cv=5).fit(X_train, y_train)

            # Predict using the model
            y_pred = lasso.predict(X_test)
            # Calculate and store the MSE value
            mse = mean_squared_error(y_test, y_pred)
            mse_values[year] = mse

            # Accumulate actual and predicted values for all years
            all_actuals.extend(y_test.tolist())
            all_predictions.extend(y_pred.tolist())

            # Get feature importance and store the top 5 features for the year
            feature_importance = np.abs(lasso.coef_)
            feature_names = X_train.columns
            features_coef = zip(feature_names, feature_importance)
            top_features = sorted(features_coef, key=lambda x: x[1], reverse=True)[:5]
            top_features_per_year[year] = top_features

        # After looping, calculate the overall MSE
        overall_mse = mean_squared_error(all_actuals, all_predictions)
        
        return overall_mse

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    six_nearest_weather_stations(40.41236,-86.94739)
    lasso_mse(40.41236,-86.94739,'USW00014835') # Purdue Airport
    station_counts = data_cleaned.groupby('STATION').size()

    #Loop all Valid Weather Stations
    #Filter weather stations if it has records from 2010 to 2023
    stations_with_enough_records = station_counts[station_counts >= 167].index
    valid_stations = data_cleaned[
        (data_cleaned['YEAR'] == 2010) & 
        (data_cleaned['STATION'].isin(stations_with_enough_records))
    ][['STATION', 'NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
    valid_stations['overall_mse'] = valid_stations.apply(lambda row: lasso_mse(row['LATITUDE'], row['LONGITUDE'], row['STATION']), axis=1)

    