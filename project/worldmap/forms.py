# forms.py

from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

class CoordinatesForm(forms.Form):
    latitude = forms.DecimalField(max_digits=9, decimal_places=6, validators=[MinValueValidator(-90.0), MaxValueValidator(90.0)])
    longitude = forms.DecimalField(max_digits=9, decimal_places=6, validators=[MinValueValidator(-180.0), MaxValueValidator(180.0)])
    lat_direction = forms.ChoiceField(label='Lat Direction', choices=[('N', 'North'), ('S', 'South')])
    long_direction = forms.ChoiceField(label='Long Direction', choices=[('E', 'East'), ('W', 'West')])

    def clean_latitude(self):
        latitude = self.cleaned_data['latitude']
        if not self._is_valid_coordinate(latitude):
            raise ValidationError(_('Invalid latitude - please enter a number.'))
        return latitude

    def clean_longitude(self):
        longitude = self.cleaned_data['longitude']
        if not self._is_valid_coordinate(longitude):
            raise ValidationError(_('Invalid longitude - please enter a number.'))
        return longitude

    def _is_valid_coordinate(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False


class TimePeriodForm(forms.Form):
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date', 'placeholder': 'dd/mm/yy'}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date', 'placeholder': 'dd/mm/yy'}))