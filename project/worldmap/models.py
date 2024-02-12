from django.db import models

# Notes: This page is used to define the structure of our database. Each class we create in this page
# represents a table in our database. 

class WeatherData(models.Model):
    STATION = models.CharField(max_length=50)
    NAME = models.CharField(max_length=200)
    LATITUDE = models.FloatField()
    LONGITUDE = models.FloatField()
    ELEVATION = models.FloatField()
    DATE = models.DateField()
    AWND = models.FloatField(null=True, blank=True)
    PRCP = models.FloatField(null=True, blank=True)
    SNOW = models.FloatField(null=True, blank=True)
    TAVG = models.FloatField(null=True, blank=True)
    TMAX = models.FloatField(null=True, blank=True)
    TMIN = models.FloatField(null=True, blank=True)

    def save(self, *args, **kwargs):
        # Convert NaN values to None before saving to the database
        for field in self._meta.fields:
            value = getattr(self, field.attname)
            if pd.isna(value):
                setattr(self, field.attname, None)
        super(WeatherData, self).save(*args, **kwargs)
