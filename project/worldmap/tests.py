import unittest
from .views import lasso_prediction
import pandas as pd
import os
import django
from project.weather import settings


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather.settings')
django.setup()

# Now we can import the Django models and any other Django-specific components


# Now you can import your Django-dependent modules
from worldmap.views import lasso_prediction
import unittest
import pandas as pd
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


class TestLassoPrediction(unittest.TestCase):
    def test_lasso_prediction(self):
        # 
        weather_stations = GlobalData.objects.all()
        data = pd.DataFrame(list(weather_stations.values()))
    
        #data = pd.DataFrame({
            #'DATE': ['2020-01-01', '2020-01-02'],
            #'SNOW': [0, 0],
            #'STATION': ['S1', 'S2'],
            #'LATITUDE': [34.05, 34.05],
            #'LONGITUDE': [-118.25, -118.25],
            #'ELEVATION': [10, 20],
            #'AWND': [5.0, 5.2],
            #'TAVG': [15, 15.5],
            #'TMAX': [20, 21],
            #'TMIN': [10, 9.8]
        #})
    
        # 
        result = lasso_prediction(data)
        # 
        self.assertIsInstance(result, pd.DataFrame)  # 
        self.assertNotEqual(len(result), 0)

# Flask application setup
from flask import Flask, jsonify, request
import pandas as pd
from project.worldmap.views import lasso_prediction


app = Flask(__name__)  # Corrected __worldmap__ to __name__

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.DataFrame(request.json)
    try:
        result = lasso_prediction(data)
        return jsonify(result.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
