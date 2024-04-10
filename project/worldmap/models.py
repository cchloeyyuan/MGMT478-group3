from django.db import models

# Notes: This page is used to define the structure of our database. Each class we create in this page
# represents a table in our database. 

class GlobalData(models.Model):
    fips_code = models.IntegerField(null=True, blank=True)
    station_id = models.CharField(max_length=255, primary_key=True)
    Latitude = models.FloatField(null=True, blank=True)
    Longitude = models.FloatField(null=True, blank=True)
    Elevation = models.FloatField(null=True, blank=True)
    Month = models.CharField(max_length=255)
    PRCP = models.FloatField(null=True, blank=True)
    AWND = models.FloatField(null=True, blank=True)
    TAVG = models.FloatField(null=True, blank=True)
    TMAX = models.FloatField(null=True, blank=True)
    TMIN = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "worldmap_GlobalData"

    def save(self, *args, **kwargs):
        # Convert NaN values to None before saving to the database
        for field in self._meta.fields:
            value = getattr(self, field.attname)
            if pd.isna(value):
                setattr(self, field.attname, None)
        super(GlobalData, self).save(*args, **kwargs)



    # Handle GET request or invalid form by rendering the page with the empty or current form
    # ...

