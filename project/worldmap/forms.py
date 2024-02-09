# forms.py

from django import forms

class CoordinatesForm(forms.Form):
    latitude = forms.FloatField(label='Latitude')
    longitude = forms.FloatField(label='Longitude')
    lat_direction = forms.ChoiceField(label='Lat Direction', choices=[('N', 'North'), ('S', 'South')])
    long_direction = forms.ChoiceField(label='Long Direction', choices=[('E', 'East'), ('W', 'West')])

