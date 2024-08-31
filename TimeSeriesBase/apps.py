from django.apps import AppConfig


class TimeseriesbaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'TimeSeriesBase'

    # def ready(self):
    #     from TimeSeriesBase.bot import run_bot_in_background
    #     run_bot_in_background()  # Start the bot when the app is ready
        