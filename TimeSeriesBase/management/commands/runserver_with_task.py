# my_app/management/commands/runserver_with_task.py

from django.core.management.commands.runserver import Command as RunserverCommand
from TimeSeriesBase.models import GrowthPredictionModel  # Import your model

class Command(RunserverCommand):
    def handle(self, *args, **options):
        # Start the background task
        file_path = "static/SampleExcel/yearSample.csv"  # Path to your data file
        model = GrowthPredictionModel()
        model.start_training_task(file_path)  # Start the background task

        # Call the original runserver command
        super().handle(*args, **options)
