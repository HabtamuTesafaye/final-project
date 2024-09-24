from django.shortcuts import render
from django. contrib import messages
from django.contrib.auth.decorators import login_required
from UserManagement.decorators import staff_user_required
from TimeSeriesBase.models import *
from django.http import JsonResponse
from django.contrib.auth.models import AnonymousUser
from django.forms.models import model_to_dict
from .decorators import public_required
from django.db.models import F
from django.shortcuts import get_object_or_404



@public_required
def index(request):
    # Fetch additional data for the context
    try:
        last_year = DataPoint.objects.filter().order_by('-year_EC')[1]
        last_last_year = DataPoint.objects.filter().order_by('-year_EC')[2]

        # CPI
        cpi_category = Category.objects.filter(name_ENG='CPI').first()
        cpi_indicators = Indicator.objects.filter(for_category=cpi_category)

        cpi_value = []
        for item in cpi_indicators:
            value_last_year = DataValue.objects.filter(for_datapoint=last_year, for_indicator=item)
            value_last_last_year = DataValue.objects.filter(for_datapoint=last_last_year, for_indicator=item)

            if value_last_year and value_last_last_year:
                sum1 = sum(val.value for val in value_last_year)
                sum2 = sum(val.value for val in value_last_last_year)
                percentage = ((sum1 - sum2) / sum2) * 100

                cpi_value.append({
                    'item': item.title_ENG,
                    'value': round(-1 * percentage if percentage < 0 else percentage, 1),
                    'link': item.id,
                    'mode': 'negative' if percentage < 0 else 'positive'
                })

        # Export Bill USD
        export_bill_category = Category.objects.filter(name_ENG='Export in Bil USD').first()
        export_bill_indicators = Indicator.objects.filter(for_category=export_bill_category)

        export_bill_value = []
        for item in export_bill_indicators:
            value_last_year = DataValue.objects.filter(for_datapoint=last_year, for_indicator=item).first()
            value_last_last_year = DataValue.objects.filter(for_datapoint=last_last_year, for_indicator=item).first()
            percentage = ((value_last_year.value - value_last_last_year.value) / value_last_last_year.value) * 100
            export_bill_value.append({
                'item': item.title_ENG,
                'value': round(-1 * percentage if percentage < 0 else percentage, 1),
                'link': item.id,
                'mode': 'negative' if percentage < 0 else 'positive'
            })

        # Context data for the template
        context = {
            'cpi': cpi_value,
            'cpi_category': cpi_category,
            'year': last_year,
            'export_bill_value': export_bill_value,
            'export_bill_category': export_bill_category,
        }
    except IndexError:
        context = {
            'cpi': [],
            'cpi_category': None,
            'year': None,
            'export_bill_value': [],
            'export_bill_category': None,
            'gdp_growth_message': "Could not retrieve additional data.",
        }

    return render(request, 'index.html', context)

@public_required
def detail_analysis(request, pk):
    return render(request, 'detail_analysis.html')

@public_required
def about(request):
    return render(request,"about.html")

@public_required
def contact(request):
    return render(request,"contact.html")


@staff_user_required
def profile_view(request):
    return render(request,"profile.html")

@public_required
def data(request):
    return render(request,"data.html")


#############################
#          JSON             #
#############################

@public_required
def json(request):
    topic = list(Topic.objects.all().values())
    year =list( DataPoint.objects.all().values())
    month_data = cache.get("month_data")
    quarter_data = cache.get("quarter_data")

    if month_data is None:
        # Fetch month data from the database if not in cache
        month_data = list(Month.objects.all().values())
        # Cache the data for future requests
        cache.set("month_data", month_data)

    if quarter_data is None:
        # Fetch quarter data from the database if not in cache
        quarter_data = list(Quarter.objects.all().values())
        # Cache the data for future requests
        cache.set("quarter_data", quarter_data)
        

    context = {
        'topics': topic,
        'year' : year,
        'quarter' : quarter_data,
        'month' : month_data,

    }
    return JsonResponse(context)

@public_required
def filter_category_lists(request,pk):
    topic = Topic.objects.get(pk = pk)
    category_lists = list(Category.objects.filter(topic = topic).prefetch_related('topic').values())
    return JsonResponse(category_lists, safe=False)

@public_required
def filter_indicator_lists(request, pk):
    category = Category.objects.get(pk = pk)
    if isinstance(request.user, AnonymousUser):
        indicators = Indicator.objects.filter(for_category = category, is_public = True).select_related("for_category")
    else:
        indicators = Indicator.objects.filter(for_category = category).select_related("for_category")

    def child_indicator_filter(parent):
        return Indicator.objects.filter(parent = parent)

    returned_json = []

    def child_list(parent, child_lists):
        for i in child_lists:
            if i.parent.id == parent.id:
                child_lists = child_indicator_filter(i)
                returned_json.extend(list(child_lists.values()))
                child_list(i,child_lists)

    returned_json.extend(list(indicators.values()))             
    for indicator in indicators:
        child_lists = child_indicator_filter(indicator)
        returned_json.extend(list(child_lists.values())) 
        child_list(indicator, child_lists)


    return JsonResponse(returned_json, safe=False)
   
@public_required
def filter_indicator_value(request, pk):
    # Use get_object_or_404 to handle the case where the category with the specified primary key does not exist
    single_category = get_object_or_404(Category, pk=pk)

    # Fetch all indicators related to the category using select_related to minimize queries
    value_new = []



    l = Indicator.objects.filter(for_category=single_category, parent=None).prefetch_related("children")

    all_indicator =  Indicator.objects.prefetch_related("children")
    returned_json = []
   
    def child_list(child_lists):
        for i in child_lists:
            child = all_indicator.filter(parent = i).prefetch_related("children")
            if child is not None:
                returned_json.extend(list(child.values('id', 'title_ENG', 'title_AMH', 'composite_key', 'op_type', 'parent_id', 'for_category_id', 'is_deleted', 'measurement_id', 'measurement__Amount_ENG', 'type_of', 'is_public')))
                child_list(child)

    returned_json.extend(list(l.values('id', 'title_ENG', 'title_AMH', 'composite_key', 'op_type', 'parent_id', 'for_category_id', 'is_deleted', 'measurement_id', 'measurement__Amount_ENG', 'type_of', 'is_public')))             
    for indicator in l:
        child_lists =all_indicator.filter(parent = indicator).prefetch_related("children")
        returned_json.extend(list(child_lists.values('id', 'title_ENG', 'title_AMH', 'composite_key', 'op_type', 'parent_id', 'for_category_id', 'is_deleted', 'measurement_id', 'measurement__Amount_ENG', 'type_of', 'is_public'))) 
        child_list(child_lists)




    # Fetch data values for each indicator in a single query
    for indicator in returned_json:
        value_filter =  DataValue.objects.filter(for_indicator__id=indicator['id']).select_related("for_datapoint", "for_indicator").values()

        for val in value_filter:
            value_new.append(val)
    return JsonResponse(value_new, safe=False)

from django.core.cache import cache
##INDEX SAMPLE DATA 
#Indicator Detail Page With Child and with Values

@public_required
def filter_indicator(request, pk):
    single_indicator = Indicator.objects.get(pk = pk)


    returned_json = []
    returned_json.append(model_to_dict(single_indicator))


    indicators = list(Indicator.objects.all().values())

    year = list(DataPoint.objects.all().values())
    
    indicator_point = list(Indicator_Point.objects.filter(for_indicator = pk).values())
    measurements = list(Measurement.objects.all().values())
    # Attempt to get data from cache
    month_data = cache.get("month_data")
    quarter_data = cache.get("quarter_data")

    if month_data is None:
        # Fetch month data from the database if not in cache
        month_data = list(Month.objects.all().values())
        # Cache the data for future requests
        cache.set("month_data", month_data)

    if quarter_data is None:
        # Fetch quarter data from the database if not in cache
        quarter_data = list(Quarter.objects.all().values())
        # Cache the data for future requests
        cache.set("quarter_data", quarter_data)
    
    indicators_with_children = Indicator.objects.filter(parent=single_indicator).prefetch_related("children")

    # Create a dictionary for each parent and child indicator
    indicator_list = [model_to_dict(single_indicator)]
    indicator_list  += [model_to_dict(child_indicator) for child_indicator in indicators_with_children]

    def child_list(parent):
        for i in indicators:
            if i['parent_id'] == parent.id:
                returned_json.append(i)
                child_list(Indicator.objects.get(id = i['id']))
                    
    
    child_list(single_indicator)

    value_new = []
    year_new = []


    # Fetch data values for each indicator in a single query
    for indicator in indicator_list:
    # Fetch DataValues and related DataPoint instances in a single query
        value_filter = DataValue.objects.filter(for_indicator__id=indicator['id']).select_related("for_datapoint", "for_indicator")
    
        for data_value in value_filter:
            for_datapoint_instance = data_value.for_datapoint
    
            # Check if the DataPoint instance is in year_new before appending
            if model_to_dict(for_datapoint_instance) not in year_new:
                year_new.append(model_to_dict(for_datapoint_instance))
    
            # Convert DataValue and DataPoint instances to dictionaries and append to value_new
            value_new.append({
                'id': data_value.id,
                'value': data_value.value,
                'for_quarter_id': data_value.for_quarter_id,
                'for_month_id': data_value.for_month_id,
                'for_datapoint_id': data_value.for_datapoint_id,
                'for_datapoint__year_EC': for_datapoint_instance.year_EC,
                'for_source_id': data_value.for_source_id,
                'for_indicator_id': data_value.for_indicator_id,
                'is_deleted': data_value.is_deleted
            })   
    
    context = {
        'indicators' :  returned_json,
        'indicator_point': indicator_point,
        'year' : year,
        'new_year' : year_new,
        'value' : value_new,
        'measurements' : measurements,
        'month': month_data,
        'quarter': quarter_data
    }
    
    return JsonResponse(context)

@public_required
def month_data(request, month_id):
    category = Category.objects.get(pk=month_id)
    child_indicators = Indicator.objects.filter(for_category=category)

    months = Month.objects.all()
    years = DataPoint.objects.all()

    data_set = []

    for child in child_indicators:
        values = DataValue.objects.filter(
            for_indicator=child,
            for_month__in=months,
            for_datapoint__in=years,
            is_deleted=False
        ).values('for_datapoint__year_EC', 'for_month__number', 'value')

        arr = [
            [
                [int(value['for_datapoint__year_EC']), int(value['for_month__number']), 1],
                value['value']
            ]
            for value in sorted(values, key=lambda x: (x['for_datapoint__year_EC'], x['for_month__number']))
        ]

        data_set.append({'name': child.title_ENG, 'data': arr})

    return JsonResponse(data_set, safe=False)

from django.db.models import F

@public_required
def quarter_data(request, quarter_id):
    category = Category.objects.get(pk=quarter_id)
    child_indicators = Indicator.objects.filter(for_category=category)

    data_set = []

    for child in child_indicators:
        values = DataValue.objects.filter(
            for_indicator=child,
            is_deleted=False
        ).values('for_datapoint__year_EC', 'for_quarter__title_ENG', 'value')

        quarters = set(value['for_quarter__title_ENG'] for value in values)
        years = sorted(set(value['for_datapoint__year_EC'] for value in values))

        arr = [
            [
                [
                    int(value['for_datapoint__year_EC']),
                    quarter_to_month(value['for_quarter__title_ENG']),
                    1
                ],
                value['value']
            ]
            for value in sorted(values, key=lambda x: (x['for_datapoint__year_EC'], quarter_to_month(x['for_quarter__title_ENG'])))
            if value['value'] is not None
        ]

        # Append data only if there is non-empty data
        if arr:
            data_set.append({'name': child.title_ENG, 'data': arr})

    return JsonResponse(data_set, safe=False)

@public_required
def quarter_to_month(quarter_title):
    # Map the quarter to perspective months
    quarter_to_month = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}
    return quarter_to_month.get(quarter_title, 1)


#####################################################
#                Prediction                         #
#####################################################

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import logging
import threading
import os
import hashlib

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Additional logging configuration for TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import numpy as np
import pandas as pd
import threading
import hashlib
import logging
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)

class GrowthPredictionModel:
    def __init__(self, file_path):
        self.model = self.build_model()
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.file_path = file_path
        self.indicator_names = []
        self.indicator_future_predictions = {}
        self.task_done = False
        self.lock = threading.Lock()
        self.file_hash = self._get_file_hash(file_path)

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(5, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(20))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def _get_file_hash(self, file_path):
        """Generate a hash of the file to detect changes."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception as e:
            logger.error(f"Error generating file hash: {e}")
        return hash_md5.hexdigest()

    def train(self, X_train, y_train, epochs=20, batch_size=16):
        original_shape = X_train.shape
        X_train = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_train_scaled = X_train_scaled.reshape(original_shape)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))  # Reshape y_train for scaler_y
        
        logger.info("Starting model training...")
        self.model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1)
        logger.info("Model training completed.")

    def predict_growth(self, X, future_steps=1):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, 5, 1)
        elif X.ndim == 2:
            X = X.reshape(1, 5, 1)

        future_predictions = []
        for step in range(future_steps):
            X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(1, 5, 1)
            prediction = self.model.predict(X_scaled)
            prediction_inverse = self.scaler_y.inverse_transform(prediction).flatten()[0]
            future_predictions.append(prediction_inverse)

            X = np.roll(X, shift=-1, axis=1)
            X[:, -1, 0] = prediction_inverse

        return future_predictions

    def read_data_from_file(self):
        try:
            # Read the CSV file
            data = pd.read_csv(self.file_path, encoding='latin1')
        except Exception as e:
            logger.error(f"Error reading the file: {e}")
            return None, None
        
        self.indicator_names = data.iloc[:, 0].values  # Store the indicator names
        data = data.iloc[:, 1:]

        # Replace non-numeric values with 0
        data = data.replace(r'^\s*$', 0, regex=True)  # Replace empty strings with 0
        data = data.replace('-', 0)  # Replace '-' with 0

        # Convert data to float
        try:
            data = data.astype(float)
        except ValueError as e:
            logger.error(f"Error converting data to float: {e}")
            return None, None

        # Drop NaN values (if any) which may arise from other errors
        if data.isna().sum().sum() > 0:
            logger.info("Data contains NaN values. Cleaning the data...")
            data = data.dropna()

        # Replace infinite values with NaN and drop them
        if np.isinf(data.values).any():
            logger.info("Data contains infinite values. Cleaning the data...")
            data = data.replace([np.inf, -np.inf], np.nan).dropna()

        return data, self.indicator_names

    def prepare_sequences(self, data, sequence_length=5):
        X, y = [], []
        for row in data.itertuples(index=False, name=None):
            row_values = np.array(row)
            if len(row_values) < sequence_length + 1:
                continue
            for i in range(len(row_values) - sequence_length):
                X_sequence = row_values[i:i + sequence_length]
                y_value = row_values[i + sequence_length]
                if pd.isna(y_value):
                    continue  # Skip this sequence if y_value is NaN
                X.append(X_sequence)
                y.append(y_value)
        
        if len(X) == 0 or len(y) == 0:
            # If no valid sequences, create dummy sequences with zeros
            X = np.zeros((1, sequence_length, 1))
            y = np.zeros((1, 1))
        else:
            X, y = np.array(X).reshape(-1, sequence_length, 1), np.array(y).reshape(-1, 1)
        
        return X, y

    def analyze_growth(self, X_train, indicator_names):
        growth_list = []
        num_indicators = len(indicator_names)
        self.indicator_future_predictions = {}  # Reset predictions

        years_intervals = [3, 5, 10, 15, 20]  # Added 15 and 20 years intervals

        for i in range(num_indicators):
            indicator = indicator_names[i]
            indicator_data = X_train[i::num_indicators]  # Slice data for each indicator
            
            # Set the initial value for each indicator as the first value in its respective column
            initial_value = indicator_data[0, 0, 0]  # Adjust this if necessary based on your data
            
            last_sequence = indicator_data[-1]
            future_predictions = self.predict_growth(last_sequence, future_steps=max(years_intervals))

            future_growths = []
            for interval in years_intervals:
                if interval <= len(future_predictions):
                    predicted_value = future_predictions[interval-1]
                    future_growths.append({
                        'years': interval,
                        'predicted_value': predicted_value,
                    })

            total_growth = future_predictions[-1]  # Use the last predicted value instead of calculating percentage
            growth_list.append(total_growth)  # Store the total growth value
            self.indicator_future_predictions[indicator] = {
                'initial_value': initial_value,
                'total_growth': total_growth,  # Store total growth value
                'future_growths': future_growths,
            }
        
        self.average_growth = np.mean(growth_list)  # Save the result for later use
        self.task_done = True  # Mark the task as done



    def process_data(self):
        current_file_hash = self._get_file_hash(self.file_path)
        if current_file_hash == self.file_hash and self.task_done:
            logger.info("Using cached data.")
            return  # Use cached data if the file has not changed and processing is done

        data, indicator_names = self.read_data_from_file()
        if data is not None and indicator_names is not None:
            X_train, y_train = self.prepare_sequences(data)
            if len(X_train) > 0:
                self.train(X_train, y_train)
                self.analyze_growth(X_train, indicator_names)
                self.file_hash = current_file_hash  # Update file hash after processing
            else:
                logger.error("No data available for training.")
        else:
            logger.error("No data available for training.")
        self.task_done = True  # Mark the task as done

    def get_indicator_future_predictions(self):
        return self.indicator_future_predictions  # Method to get future growth predictions for each indicator

    def is_task_done(self):
        return self.task_done  # Method to check if the background task is done

    def start_background_task(self):
        def task():
            with self.lock:
                self.process_data()
        background_thread = threading.Thread(target=task, daemon=True)
        return background_thread


FILE_PATH = 'static/SampleExcel/yearSample.csv'
model = GrowthPredictionModel(FILE_PATH)

def start_growth_analysis(request):
    global model  # Use the global model instance
    
    # Start the background task
    background_thread = model.start_background_task()
    background_thread.start()
    message = 'Background process started. Please wait a few moments and check the result.'

    return JsonResponse({'message': message, 'flag': True})

def get_indicator_future_predictions(request):
    if not model.is_task_done():
        return JsonResponse({'flag': False, 'error': 'Background task is still running. Please try again later.'})

    indicator_future_predictions = model.get_indicator_future_predictions()
    if not indicator_future_predictions:
        return JsonResponse({'flag': False, 'error': 'No predictions available. Please try again later.'})

    # Convert NumPy float32 values to Python float and prepare detailed response
    converted_predictions = {}
    for indicator, prediction in indicator_future_predictions.items():
        converted_predictions[indicator] = {
            'initial_value': float(prediction['initial_value']),
            'future_growths': [
                {
                    'years': growth['years'],
                    'predicted_value': float(growth['predicted_value'])
                } for growth in prediction['future_growths']
            ]
        }

    # Generate AI insights
    ai_insights = generate_ai_insights(converted_predictions)

    return JsonResponse({
        'flag': True,
        'indicatorFuturePredictions': converted_predictions,
        'aiInsights': ai_insights
    })

import requests
import json as toJSON
import time

def generate_ai_insights(data):
    api_key = ''  # Use environment variable for API key
    if not api_key:
        return "API key not found."

    api_url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    # Adjust the prompt to instruct the AI to generate a structured report with paragraphs
    prompt = (
        "Generate a summary about this prediction data, including insights and recommendations. "
        "Here is the data:\n\n"
        f"{toJSON.dumps(data, indent=2)}"
    )
    
    payload = {
        'contents': [
            {
                'parts': [
                    {'text': prompt}
                ]
            }
        ]
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=payload)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            ai_response = response.json()

            # Extract the AI-generated text
            candidates = ai_response.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts:
                    paragraphs = [part.get('text', '') for part in parts if part.get('text')]
                    if paragraphs:
                        return '\n\n'.join(paragraphs)  # Join paragraphs with double newlines
            return "No content returned from API."

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error occurred: {http_err}")
                return f"Failed to generate report from AI API: {http_err}"
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            return "Failed to generate report from AI API."
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            return "Failed to generate report from AI API."
        except requests.exceptions.RequestException as req_err:
            print(f"Error occurred: {req_err}")
            return "Failed to generate report from AI API."

    return "Failed to generate report after multiple attempts. Please try again later."

@csrf_exempt
def upload_csv(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'flag': False, 'error': 'No file uploaded.'})
        
        file = request.FILES['file']
        file_path = os.path.join('media', 'uploaded_file.csv')
        
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        global model
        model.file_path = file_path
        model.file_hash = model._get_file_hash(file_path)  # Reset file hash
        model.task_done = False  # Reset task status
        return JsonResponse({'flag': True, 'message': 'File uploaded successfully.'})
