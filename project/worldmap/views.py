# views.py
from django.shortcuts import render
import folium
import sys
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
from shapely.geometry import Polygon, MultiPolygon
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
from shapely import wkt


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
    
    # Add marker for each unique weather station with average values
    #for index, row in station_averages.iterrows():
        #popup_text = f"{row['station_id']}<br>Avg TMIN: {row['TMIN']}°C<br>Avg TMAX: {row['TMAX']}°C<br>Avg PRCP: {row['PRCP']}(mm)"
        #folium.Marker([row['Latitude'], row['Longitude']], popup=popup_text).add_to(my_map)


    #Read the statis csv file in. Then convert geometry to be ready to switch to geodataframe. 
    heatmap_df = pd.read_csv("heatmap_results_final.csv")

    heatmap_df['geometry'] = heatmap_df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(heatmap_df, geometry='geometry')
    gdf.crs = "EPSG:4326"

    # # Create Folium map
        # Add choropleth layer to the map
    folium.Choropleth(
        geo_data=gdf,
        name='Precipitation',
        data=gdf,
        columns=['GEOID', 'PRCP Measure'],
        key_on='feature.properties.GEOID',
        fill_color='YlOrRd',  # Change the color scale if needed
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Average Precipitation (mm)'
    ).add_to(my_map)

    folium.Choropleth(
        geo_data=gdf,
        name='Temperature',
        data=gdf,
        columns=['GEOID', 'TAVG Measure'],
        key_on='feature.properties.GEOID',
        fill_color='YlOrRd',  # Change the color scale if needed
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Average Temperature (C)'
    ).add_to(my_map)



    style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}

    highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}
    
    NIL = folium.features.GeoJson(
        gdf,
        style_function=style_function, 
        control=False,
        highlight_function=highlight_function, 
        tooltip=folium.features.GeoJsonTooltip(
            fields=['NAME','PRCP Measure','TAVG Measure'],
            aliases=['County Name: ','Precipitation average in mm:  ','Average Temperature in C: '],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
        )
    )

    my_map.add_child(NIL)
    my_map.keep_in_front(NIL)
    
    # Add dark and light mode. 
    folium.TileLayer('cartodbdark_matter',name="dark mode",control=True).add_to(my_map)
    folium.TileLayer('cartodbpositron',name="light mode",control=True).add_to(my_map)
    
    # Add layer control
    folium.LayerControl().add_to(my_map)
    # Convert the Folium map to HTML
    map_html = my_map._repr_html_()

    return render(request, 'map.html', {'map_html': map_html})

import logging
logger = logging.getLogger(__name__)

def combined_request(request):
    context = {}
    try:
        if request.method == 'POST':
            form_type = request.POST.get('form_type')

            # Handle Geographic Location Form
            if form_type == 'geo_location':
                geo_form = CoordinatesForm(request.POST)
                if geo_form.is_valid():
                    # Process the geographic data
                    latitude = geo_form.cleaned_data['latitude']
                    longitude = geo_form.cleaned_data['longitude']
                    lat_direction = geo_form.cleaned_data['lat_direction']
                    long_direction = geo_form.cleaned_data['long_direction']

                    latitude = latitude if lat_direction == 'N' else -latitude
                    longitude = longitude if long_direction == 'E' else -longitude

                    my_map = folium.Map(location=[float(latitude), float(longitude)], zoom_start=14)
                    folium.Marker([float(latitude), float(longitude)], popup="Your Location").add_to(my_map)
                    context['map_html'] = my_map._repr_html_()
                else:
                    context['geo_form'] = geo_form

            # Handle Date Range Form
            elif form_type == 'date_range':
                date_form = TimePeriodForm(request.POST)
                if date_form.is_valid():
                    # Process the date range data
                    start_date = date_form.cleaned_data['start_date']
                    end_date = date_form.cleaned_data['end_date']

                    data_for_period = GlobalData.objects.filter(date_recorded__gte=start_date, date_recorded__lte=end_date)
                    context['data'] = data_for_period
                else:
                    context['date_form'] = date_form

        # Ensure both forms are always in context, either as new or with errors
        context.setdefault('geo_form', CoordinatesForm())
        context.setdefault('date_form', TimePeriodForm())

        return render(request, 'map.html', context)
    
    except Exception as e:
        logger.error(f"An error occurred while processing the form:{str(e)}")
        raise

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
            return HttpResponse(status=204)  # No content to return
        except Exception as e:
            # If fail use JsonResponse to return message
            return HttpResponse(status=204)

    return HttpResponse(status=204)

    #map_html = my_map._repr_html_()
    #return render(request, 'map.html', {'form': form, 'map_html': map_html})

def heatmap(station_averages, county_coords, color_value):

    weighted_values = []
    if isinstance(county_coords, Polygon) or isinstance(county_coords, MultiPolygon):
        # Extract all coordinates regardless of Polygon or MultiPolygon
        if isinstance(county_coords, Polygon):
            all_coordinates = county_coords.exterior.coords
        elif isinstance(county_coords, MultiPolygon):
            all_coordinates = []
            for poly in county_coords.geoms:
                all_coordinates.extend(poly.exterior.coords)

    # Extract latitude and longitude coordinates
    latitudes = [coord[1] for coord in all_coordinates]
    longitudes = [coord[0] for coord in all_coordinates]
    # Fit KNN model on station coordinates
    warnings.filterwarnings("ignore", category=UserWarning)
    knn_model = NearestNeighbors(n_neighbors=3).fit(station_averages[['Latitude', 'Longitude']])

    # Find indices of the 3 closest stations for each grid point
    _, indices = knn_model.kneighbors(np.column_stack((latitudes, longitudes)))
    for index_set in indices:
        # Get the distances to the closest stations
        distances = knn_model.kneighbors()[0]
        # Calculate the weights based on the inverse of distance
        weights = 1 / (distances+1.65)

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

def update_precipitation(station_averages):
    null_prcp_stations = station_averages[station_averages['PRCP'].isnull()]

    # Delete rows with null values for 'PRCP'
    if not null_prcp_stations.empty:
        print("Stations with null values for Precipitation (PRCP):")
        print(null_prcp_stations)
        # Drop rows with null values for 'PRCP'
        station_averages.dropna(subset=['PRCP'], inplace=True)
        print("Rows with null values for Precipitation (PRCP) have been deleted.")
    else:
        print("No stations found with null values for Precipitation (PRCP).")

    heatmap_df = gpd.read_file("counties.geojson")
    heatmap_df['PRCP Measure'] = 0 #This creates a new column to store values for the heatmap
    undesired_state_codes = ["08", "48", "37"]  # These are the states that don't have any stations (Texas, Colorado, and North Carolina)
    desired_color_value = "PRCP" # This is the column name that is used for the values of the heatmap

    # Loop through each county and apply the heatmap knn function to give it a measure value
    for index, row in heatmap_df.iterrows():
        if row['STATEFP'] not in undesired_state_codes:
            measure = heatmap(station_averages, row['geometry'], desired_color_value)
            rounded_measure = round(measure, 2)
            heatmap_df.loc[index, 'PRCP Measure'] = rounded_measure
        else:
            heatmap_df.drop(index, inplace=True)

    heatmap_df = heatmap_df.reset_index(drop=True)
    # Read in existing prediction file and assign index to 'NAME' column to update prediction values
    heatmap_df_existing = pd.read_csv("heatmap_results_final.csv")

    # Update 'TAVG Measure' column in heatmap_df_existing with values from heatmap_df
    heatmap_df_existing['PRCP Measure'] = heatmap_df['PRCP Measure']

    # Reset the index to make 'NAME' a regular column again
    heatmap_df_existing.to_csv("heatmap_results_final.csv", index=False)

    return heatmap_df_existing

def update_temperature(station_averages):
    null_temp_stations = station_averages[station_averages['TAVG'].isnull()]

    # Delete rows with null values for 'PRCP'
    if not null_temp_stations.empty:
        print("Stations with null values for Temperature (TAVG):")
        print(null_temp_stations)
        # Drop rows with null values for 'PRCP'
        station_averages.dropna(subset=['TAVG'], inplace=True)
        print("Rows with null values for Temperature (TAVG) have been deleted.")
    else:
        print("No stations found with null values for Temperature (TAVG).")

    heatmap_df = gpd.read_file("counties.geojson")
    heatmap_df['TAVG Measure'] = 0 #This creates a new column to store values for the heatmap
    undesired_state_codes = ["08", "48", "37"]  # These are the states that don't have any stations (Texas, Colorado, and North Carolina)
    desired_color_value = "TAVG" # This is the column name that is used for the values of the heatmap

    # Loop through each county and apply the heatmap knn function to give it a measure value
    for index, row in heatmap_df.iterrows():
        if row['STATEFP'] not in undesired_state_codes:
            measure = heatmap(station_averages, row['geometry'], desired_color_value)
            rounded_measure = round(measure, 2)
            heatmap_df.loc[index, 'TAVG Measure'] = rounded_measure
        else:
            heatmap_df.drop(index, inplace=True)

    heatmap_df = heatmap_df.reset_index(drop=True)
    # Read in existing prediction file and assign index to 'NAME' column to update prediction values
    heatmap_df_existing = pd.read_csv("heatmap_results_final.csv")

    # Update 'TAVG Measure' column in heatmap_df_existing with values from heatmap_df
    heatmap_df_existing['TAVG Measure'] = heatmap_df['TAVG Measure']

    # Reset the index to make 'NAME' a regular column again
    heatmap_df_existing.to_csv("heatmap_results_final.csv", index=False)

    return heatmap_df_existing

def update_predictions_request(request):
    update_predictions()
    data = {
        'success': True,
        'message': 'Predictions updated successfully.'
    }
    return JsonResponse(data)

def update_predictions():
    # Get weather station data from the model
    weather_stations = GlobalData.objects.all()
    # Convert the QuerySet to a Pandas DataFrame
    df = pd.DataFrame(list(weather_stations.values()))
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
    print("station averages was created")
    # Right now this is the code to update precipitation predicted values
    update_temperature(station_averages)
    update_precipitation(station_averages)
    return