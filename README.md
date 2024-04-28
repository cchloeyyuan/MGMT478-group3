# Weather Data Analysis Server
## Group 3 - Rhett Offenbacker, Wei Yuan, Caleb Hammoudeh, Gabriel Chang

## Overview
This project is part of the MGMT 478 course and aims to create a Weather Data Analysis Server. The server utilizes data from a MySQL database, applies prediction algorithms (KNN nearest neighbor), visualizes the predictions using the Folium library and a Choropleth heatmap coloring layer, and presents the results on an HTML/JavaScript web page.

## Features
* Data Retrieval: The server retrieves historical weather station data from a MySQL database.
* Prediction Algorithms: Custom prediction algorithms are applied to the retrieved data in order to predict weather values like temperature and precipitation for a given county in the United States.
* Visualization: Predictions are visualized using the Folium library to create interactive heatmaps. It will detail the predicted temperature and precipitation value for each county in the United States using a color gradient. This excludes 3 states (Colorado, Texas, and North Caroline) because we have no weather station data from these states. 
* Web Interface: All functionalities and visualizations are accessible through an HTML/JavaScript web page. In addition, the website has many functionalities that are listed below:

## File Structure
* worldmap/models.py: This python script is used to create the weather station model for our data. This is crucial as it reads in the table columns creates a dataframe we can use based on the data in mySQL.
* worldmap/views.py: This contains all of our code that is ran when the server is initiated.
  * def map_view(): This is our main function that runs at server startup:
    1. It creates a dataframe so we can access the historical weather station data.
    2. It reads our local csv file that has our prediction values for temperature and precipitation stored for each county.
    3. It converts the csv file to a geodataframe, then we use the folium.Choropleth library to create a heatmap color gradient based on the values for temperature and precipitation for each county.
    4. It adds the heatmap layers to our folium map and creates a hovering system where if you put your mouse over a given county it will show the county name and predicted temperature/precipitation values.
    5. Then all of our heatmap layers and features are added to our final map which is rendered into our map.html website page.
  * def heatmap(): This function is where our predicted values are calculated for each county.
    1. The function reads in all of the coordinates for a county boundary.
    2. It uses our weather station historical dataset and finds the 3 closest weather stations to a given county using KNN nearest neighbors.
    3. It weights those stations temperature or precipitation value based on their distance from the county. Then you sum up each value together to get the official predicted value.
  * def update_precipitation(): This function is used to read in our counties.geojson file and predicting precipitation values for each county and storing them in the local csv file called heatmap_results_final.
  * def update_temperature(): This function is used to read in our counties.geojson file and predicting temperature values for each county and storing them in the local csv file called heatmap_results_final.
  * def update_predictions_request(): This function handles the request from the website when a user presses on the update prediction values button. It takes the request and lets another function called update_predictions() know to update the prediction values for each county.
* worldmap/templates/map.html: This file contains all of the html and javascript code we have used to build our website. 
* weather/settings.py: This file is where we can adjust the settings of our entire Django environment. This is where we added our mySQL server information in order to connect to the mySQL database when we run our server. 
* weather/urls.py: This file is where we establish the different ways to associate the Django server code with our hmtl website. For example, we have a designated url for when our update prediction values button is pressed and it uses the url to know which prediction function to run in our worldmap/views.py file.
* heatmap_results_final.csv: This is the file that stores the predicted precipitation and temperature values for each county. This is read at initial server startup.
* counties.geojson: This file stores details about every county in the United States which includes a polygon shape of coordinates which allows us to visualize outlines around each of the counties on our folium map. 
