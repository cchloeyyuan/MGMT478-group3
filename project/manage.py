#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import pandas as pd
from worldmap.models import WeatherData

# Replace the file path with your actual file path
file_path = "C:\\Users\\caleb\\OneDrive\\Desktop\\MGMT478-group3\\Bloomington Weather Data.csv"

# Read CSV into a DataFrame
df = pd.read_csv(file_path)

# Convert DataFrame to a list of dictionaries and create WeatherData objects
data_to_insert = df.to_dict(orient='records')
WeatherData.objects.bulk_create([WeatherData(**data) for data in data_to_insert])



def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
