# Generated by Django 5.0.1 on 2024-01-31 03:49

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="WeatherData",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("STATION", models.CharField(max_length=50)),
                ("NAME", models.CharField(max_length=100)),
                ("LATITUDE", models.FloatField()),
                ("LONGITUDE", models.FloatField()),
                ("ELEVATION", models.FloatField()),
                ("DATE", models.DateField()),
                ("DAPR", models.FloatField(blank=True, null=True)),
                ("MDPR", models.FloatField(blank=True, null=True)),
                ("PRCP", models.FloatField(blank=True, null=True)),
                ("SNOW", models.FloatField(blank=True, null=True)),
                ("SNWD", models.FloatField(blank=True, null=True)),
                ("TMAX", models.FloatField(blank=True, null=True)),
                ("TMIN", models.FloatField(blank=True, null=True)),
                ("TOBS", models.FloatField(blank=True, null=True)),
                ("WT01", models.BooleanField(default=False)),
                ("WT03", models.BooleanField(default=False)),
                ("WT04", models.BooleanField(default=False)),
                ("WT05", models.BooleanField(default=False)),
                ("WT06", models.BooleanField(default=False)),
                ("WT11", models.BooleanField(default=False)),
            ],
        ),
    ]