from django.db import models
import pandas as pd

# Create your models here.
class WeatherData(models.Model):
    STATION = models.CharField(max_length=50)
    NAME = models.CharField(max_length=100)
    LATITUDE = models.FloatField()
    LONGITUDE = models.FloatField()
    ELEVATION = models.FloatField()
    DATE = models.DateField()
    DAPR = models.FloatField(null=True, blank=True)
    MDPR = models.FloatField(null=True, blank=True)
    PRCP = models.FloatField(null=True, blank=True)
    SNOW = models.FloatField(null=True, blank=True)
    SNWD = models.FloatField(null=True, blank=True)
    TMAX = models.FloatField(null=True, blank=True)
    TMIN = models.FloatField(null=True, blank=True)
    TOBS = models.FloatField(null=True, blank=True)
    WT01 = models.BooleanField(default=False)
    WT03 = models.BooleanField(default=False)
    WT04 = models.BooleanField(default=False)
    WT05 = models.BooleanField(default=False)
    WT06 = models.BooleanField(default=False)
    WT11 = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        # Convert NaN values to None before saving to the database
        for field in self._meta.fields:
            value = getattr(self, field.attname)
            if pd.isna(value):
                setattr(self, field.attname, None)
        super(WeatherData, self).save(*args, **kwargs)