# my_app/management/commands/start_growth_prediction.py

from django.core.management.base import BaseCommand
from TimeSeriesBase.models import GrowthPredictionModel  # Import your model

class Command(BaseCommand):
    help = 'Start the background task for GrowthPredictionModel'

    def handle(self, *args, **options):
        file_path = "static/SampleExcel/yearSample.csv"  # Path to your data file
        model = GrowthPredictionModel()
        model.start_training_task(file_path)  # Start the background task
        self.stdout.write(self.style.SUCCESS('Training task started in the background.'))
