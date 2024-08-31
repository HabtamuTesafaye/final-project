from django.db.models import Count
from django.shortcuts import render
from django.http import JsonResponse
from .serializers import DashboardTopicSerializer 
from TimeSeriesBase.models import DashboardTopic , Category, DataValue , Indicator , DataPoint, Month, Quarter
from django.db.models import Q
from rest_framework.decorators import api_view
import time
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.forms.models import model_to_dict
from UserManagement.models import CustomUser
from UserManagement.forms import LoginFormDashboard , PasswordChangeForm , EditProfileForm
from  UserManagement.decorators import dashboard_user_required
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth import login,authenticate,logout
from django.contrib.auth.hashers import make_password
import json
from django.http import JsonResponse , HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import os
import json
from django.http import JsonResponse
from django.views.decorators.http import require_GET


############################################
#                   payment                #
############################################
# Path to your JSON file
JSON_FILE_PATH = r'static/SampleExcel/db.json'

@require_GET
def serve_json_data(request):
    try:
        # Read the JSON file
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        # Return the JSON data as a response
        return JsonResponse(json_data, safe=False)

    except FileNotFoundError:
        return JsonResponse({'error': 'File not found'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Error decoding JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
@csrf_exempt
def dashboard_register(request, filename="static/SampleExcel/db.json"):
    if request.method == 'POST':
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        password = request.POST.get('password')
        re_password = request.POST.get('re_password')

        # Check if the password is empty
        if not password:
            return JsonResponse({'message': 'Password cannot be empty'}, status=400)

        # Validate passwords match
        if password != re_password:
            return JsonResponse({'message': 'Passwords do not match'}, status=400)

        # Check if the email exists in the CustomUser model
        if CustomUser.objects.filter(email=email).exists():
            return JsonResponse({'message': 'Email already registered in our system'}, status=400)

        # Load existing data from JSON file
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = []
        except json.JSONDecodeError:
            data = []

        # Check if the email is already registered in the JSON file
        if any(user.get('email') == email for user in data):
            return JsonResponse({'message': 'Email already registered in our system'}, status=400)

        # Find the maximum numeric part in the existing IDs
        max_num = 0
        for user in data:
            if 'id' in user and user['id'].startswith('waitingusr'):
                num_part = int(user['id'][10:])  # Extract the numeric part
                if num_part > max_num:
                    max_num = num_part

        # Create the new user data with the next available ID
        new_id = f'waitingusr{max_num + 1:03d}'  # Format to match the pattern

        user_data = {
            'id': new_id,
            'email': email,
            'first_name': first_name,
            'last_name': last_name,
            'password': make_password(password),  # In a real application, never store plaintext passwords
        }

        # Append the new user data
        data.append(user_data)

        # Write the updated data back to the file
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        # Save the user data in the session
        request.session['user_data'] = user_data
        return JsonResponse({'redirect_url': '/dashboard-api/payment/'})

    return render(request, 'dashboard-pages/authentication/register.html')

from django.templatetags.static import static
from django.core.mail import EmailMessage
from django.utils.html import format_html
from django.conf import settings
from django.templatetags.static import static
from django.utils import timezone

def payment_callback(request):
    if request.method == 'GET':
        print("GET Request")
        status = request.GET.get('status')
        tx_ref = request.GET.get('trx_ref')
        
        if status == 'success':
            print(f"Payment successful for transaction: {tx_ref}")
            
            try:
                user_email = None
                user_info = None

                # Read the JSON file
                with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)

                    # Locate the user data associated with the transaction reference
                    for user in json_data:
                        if user.get('Transaction_Id') == tx_ref:
                            user_email = user.get('email')
                            user_fname = user.get('first_name')
                            user_lname = user.get('last_name')
                            hashed_password = user.get('password')
                            payment_amount = user.get('Amount')
                            payment_deadline = user.get('Payment_Deadline')
                            start_date = user.get('Start_at')
                            user_info = user
                            break

                if user_email:
                    # Create the user
                    user = CustomUser.objects.create(
                        email=user_email,
                        first_name=user_fname,
                        last_name=user_lname,
                        username=user_fname,
                        password=hashed_password,
                        is_dashboard=True
                    )
                    print("User created successfully")

                    # Update the JSON file with registration status
                    if user_info:
                        user_info['status'] = 'paid'
                        user_info['registered'] = 'successfully'
                        user_info['Start_at'] = timezone.now().strftime('%Y-%m-%d')
                        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as file:
                            json.dump(json_data, file, ensure_ascii=False, indent=4)

                    # Construct the static URL for the logo
                    for user in json_data:
                        if user.get('Transaction_Id') == tx_ref:
                            start_date = user.get('Start_at')
                            user_info = user
                            break
                    # Send the welcome email with payment details
                    try:
                        subject = "Welcome to Our Service"
                        # Define the HTML email body with the embedded logo
                        body = format_html(f"""
                            <html>
                                <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
                                    <div style="text-align: center; padding: 20px;">
                <img src="file:///C:/Users/Bethel/OneDrive/Desktop/final%20project/Time-Series-Data/static/assets/image/photo_2023-11-09_22-23-27.jpg" alt="Company Logo" style="width: 150px; height: auto;"/>
                                    </div>
                                    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                                        <p>Dear {user_fname},</p>
                                        <p>Thank you for registering with us and successfully making a payment of <strong>{payment_amount}</strong>.</p>
                                        <p>Your subscription starts on <strong>{start_date}</strong> and will last until <strong>{payment_deadline}</strong>. Your next payment is due by <strong>{payment_deadline}</strong>.</p>
                                        <p>We are excited to have you on board!</p>
                                        <p>Best regards,<br>Ministry of Plan and development</p>
                                    </div>
                                </body>
                            </html>
                        """)

                        email = EmailMessage(
                            subject=subject,
                            body=body,
                            from_email=settings.EMAIL_HOST_USER,
                            to=[user_email],
                        )
                        email.content_subtype = 'html'  # This is crucial for sending HTML emails
                        email.send(fail_silently=False)
                        print("Welcome email with payment details and logo sent successfully")

                    except Exception as e:
                        print(f"Error sending welcome email: {e}")

                    return redirect('dashboard-login')

                else:
                    print(f"No user found with tx_ref {tx_ref}")
                    return JsonResponse({'message': 'User not found'}, status=404)

            except Exception as e:
                print(f"Error processing payment callback: {e}")
                return JsonResponse({'message': 'Internal server error'}, status=500)

        else:
            print(f"Payment failed. Status: {status}")
            return JsonResponse({'message': 'Payment failed or not successful'}, status=400)

    return JsonResponse({'message': 'Invalid Request'}, status=400)

JSON_FILE_PATH = r'static/SampleExcel/db.json'

@csrf_exempt
def update_unique_id(request):
    print("Started processing the request")
    if request.method == 'POST':
        print("Received POST request")
        try:
            # Parse request body to JSON
            data = json.loads(request.body)
            email = data.get('email')
            unique_id = data.get('uniqueId')
            Deadline = data.get('expirationDate')
            Total_amount = data.get('zamount')

            print(f"Deadline: {Deadline}, Unique ID: {unique_id}")
            print(f"Email: {email}, Unique ID: {unique_id}")

            # Check if email and unique_id are provided
            if not email or not unique_id:
                print("Missing email or uniqueId")
                return HttpResponseBadRequest("Missing email or uniqueId.")

            # Read the JSON file
            try:
                with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                   
                    
            except FileNotFoundError:
                print("JSON file not found.")
                return HttpResponseBadRequest("JSON file not found.")
            except json.JSONDecodeError:
                print("Error decoding JSON file.")
                return HttpResponseBadRequest("Error decoding JSON file.")

            # Iterate over the array to find the user by email and update their uniqueId
            user_updated = False
            for user in json_data:
                if user.get('email') == email:
                    user['Transaction_Id'] = unique_id
                    user['Payment_Deadline'] = Deadline
                    user['Amount'] = Total_amount
                    user_updated = True
                    print(f"Updated user {email} with uniqueId {unique_id}")
                    break

            if not user_updated:
                print("User not found.")
                return HttpResponseBadRequest("User not found.")

            # Write the updated data back to the file
            try:
                with open(JSON_FILE_PATH, 'w', encoding='utf-8') as file:
                    json.dump(json_data, file, ensure_ascii=False, indent=2)
            except IOError as e:
                print(f"Error writing to JSON file: {e}")
                return HttpResponseBadRequest(f"Error writing to JSON file: {e}")

            return JsonResponse({"status": "success", "uniqueId": unique_id})
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return HttpResponseBadRequest(f"An unexpected error occurred: {str(e)}")
    else:
        print("Invalid request method")
        return HttpResponseBadRequest("Invalid request method.")

def return_type(request):
    if request.method == 'POST':
        return JsonResponse({"message": "Post performed"})
    else:
        return redirect('dashboard-login')

def payment(request):
    user_data = request.session.get('user_data', {})
    if not user_data:
        return redirect('dashboard_register')

    if request.method == 'POST':
        print("Hello")
        # Hash the password and create the user
        hashed_password = make_password(user_data['password'])
        user = CustomUser.objects.create(
            email=user_data['email'],
            first_name=user_data['first_name'],
            last_name=user_data['last_name'],
            username=user_data['first_name'],
            password=hashed_password,
            is_dashboard=True
        )
        print("User created")
        
        # Clear the user_data from the session after successful registration
        return redirect('dashboard-login')

    return render(request, 'dashboard-pages/payment.html', {'user_data': user_data})

def get_user_data(email):
    try:
        # Read the JSON file
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        # Search for the user's data
        for record in json_data:
            if record.get('email') == email:
                return record  # Return the user record if found

        return None  # Return None if no record is found

    except FileNotFoundError:
        raise FileNotFoundError('File not found')
    except json.JSONDecodeError:
        raise ValueError('Error decoding JSON')
    except Exception as e:
        raise Exception(str(e))

def payment_history(request):
    user_email = request.user.email
    print(user_email)
    user_data = get_user_data(user_email)
    print(user_data)
    return render(request, 'dashboard-pages/payment_history.html' , {'user_data': user_data})

def dashboard_profile(request):
    return render(request,'dashboard-pages/authentication/profile.html')

from django.contrib.auth import update_session_auth_hash

def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_first_time = False
            user.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Successfully Updated!')
    else:
        form = PasswordChangeForm(request.user)
    return render(request,'dashboard-pages/authentication/change_pass.html', {'form': form})

def profile_updated(request):
    user = CustomUser.objects.get(pk = request.user.pk)
    form = EditProfileForm(request.POST or None, request.FILES or None,instance=user)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Updated!')
        else:
            messages.error(request, 'Please tye again!')  

    context = {
        'form' : form
    }       
    return render(request, 'dashboard-pages/authentication/edit_profile.html', context)


###########################################
#                  dashbord               #
###########################################
from django.utils import timezone
from django.contrib import messages
import json
import os

def dashboard_login(request):
    form = LoginFormDashboard(request.POST or None)
    if form.is_valid():
        email = form.cleaned_data['email']
        password = form.cleaned_data['password']
        
        # Check if the user exists in the Django database
        user = authenticate(request, email=email, password=password)
        if user is not None and user.is_dashboard:
            login(request, user)
            return redirect('dashboard-index')
        
        # If user is not found in the Django database, check in the db.json file
        else:
            if os.path.exists(JSON_FILE_PATH):
                try:
                    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
                        json_data = json.load(file)
                        
                        user_data = next((item for item in json_data if item['email'] == email), None)
                        if user_data:
                            # Check if the account has expired
                            expire_date_str = user_data.get('expire_date')
                            if expire_date_str:
                                expire_date = timezone.datetime.strptime(expire_date_str, '%Y-%m-%d').date()
                                today = timezone.now().date()
                                
                                if today > expire_date:
                                    # Update the user record in the Django database
                                    user_in_db = CustomUser.objects.filter(email=email).first()
                                    if user_in_db:
                                        user_in_db.is_dashboard = False
                                        user_in_db.save()
                                        messages.error(request, 'Your account has expired.')
                                        return render(request, 'dashboard-pages/authentication/login.html', {'form': form})
                            
                           
                            # If user is found in JSON file but not in Django database, redirect to payment page
                            request.session['user_data'] = user_data
                            return redirect('payment')
                        
                        # If email is not found in JSON file either
                        messages.error(request, 'Invalid email or password.')
                        
                except json.JSONDecodeError:
                    messages.error(request, 'Database error: db.json is corrupt.')
            else:
                messages.error(request, 'Database error: db.json not found.')
        
    context = {
        'form': form
    }
    return render(request, 'dashboard-pages/authentication/login.html', context=context)

@login_required(login_url='dashboard-login')
@dashboard_user_required
def index(request):
    return render(request, 'dashboard-pages/dashboard-index.html')

@login_required(login_url='dashboard-login')
def dashboard_logout(request):
    logout(request)
    return redirect('dashboard-login')

@login_required(login_url='dashboard-login')
@dashboard_user_required
@api_view(['GET'])


    
def pie_chart_data(request):

    if request.method == 'GET':
        topics = list(DashboardTopic.objects.annotate(category_count=Count('category')).select_related().values('id','title_ENG' , 'category_count'))
        topics_id_list = DashboardTopic.objects.annotate(category_count=Count('category'))
        category = list(Category.objects.filter(Q(dashboard_topic__in=topics_id_list)).annotate(category_count=Count('indicator')).select_related().values('id' ,'name_ENG' , 'category_count' , 'dashboard_topic__id'))
        category_id_list = list(Category.objects.filter().values_list('id', flat = True))
        indicators = list(Indicator.objects.filter(Q(for_category__id__in=category_id_list)).values(
    'id',
    'title_ENG',
    'for_category__id',
))

           
           
   
        context = {
            "topics" : topics,
            "category" : category,
            "indicators" : indicators,
            
        }
        
        return JsonResponse(context)



@login_required(login_url='dashboard-login')
@dashboard_user_required
@api_view(['GET'])
def topic_lists(request):

    if request.method == 'GET':
        topics = DashboardTopic.objects.annotate(category_count=Count('category')).select_related().order_by('order')
        # topics = topics.filter(~Q(category_count = 0)) #Only Display with category > 0
        serializer = DashboardTopicSerializer(topics, many=True)
        
        return JsonResponse({'topics':serializer.data})
    


@login_required(login_url='dashboard-login')
@dashboard_user_required
@api_view(['GET'])
def category_list(request , id): 
               
        indicator_list_id = list(Category.objects.filter(dashboard_topic__id = id).prefetch_related('indicator__set').all().values_list('indicator__id', flat=True))

        value_filter = list(DataValue.objects.filter( Q(for_indicator__id__in=indicator_list_id) & ~Q(for_datapoint_id__year_EC = None)).select_related("for_datapoint", "for_indicator").values(
            'for_indicator__type_of',
            'value',
            'for_indicator_id',
            'for_datapoint_id__year_EC',
            'for_datapoint_id__year_GC',
            'for_quarter_id',
            'for_month_id__month_AMH',
            
        ))
        

        queryset = Category.objects.filter(dashboard_topic__id = id).prefetch_related('indicator__set').filter(indicator__is_dashboard_visible = True).values(
                'dashboard_topic__title_ENG',
                'id',
                'name_ENG',
                'name_AMH',
                'indicator__id',
                'indicator__title_ENG',
                'indicator__title_AMH',
                'indicator__is_dashboard_visible',
                'indicator__type_of'   
            )
        
        categories_lists = Category.objects.filter(dashboard_topic__id = id).annotate(count_indicators=Count('indicator')).filter(indicator__is_dashboard_visible = True).values(
                'id',
                'name_ENG',
                'name_AMH', 
                'count_indicators'
            )
        categories_lists = categories_lists.filter(~Q(count_indicators = 0)).values(
            'id',
            'name_ENG',
            'name_AMH', 
            'count_indicators'
        )

        
        
        if 'q' in request.GET:
            q = request.GET['q']
            dashboard_topic = DashboardTopic.objects.all()
            queryset = Category.objects.filter().prefetch_related('indicator__set').filter(Q(indicator__title_ENG__contains=q,dashboard_topic__in = dashboard_topic ) | Q(indicator__for_category__name_ENG__contains=q ,dashboard_topic__in = dashboard_topic) ).values(
                'dashboard_topic__title_ENG',
                'id',
                'name_ENG',
                'name_AMH',
                'indicator__id',
                'indicator__title_ENG',
                'indicator__title_AMH',
                'indicator__is_dashboard_visible',
                'indicator__type_of'
            )
            indicator_list_id = queryset.values_list('indicator__id', flat=True)

            value_filter = list(DataValue.objects.filter( Q(for_indicator__id__in=indicator_list_id) & ~Q(for_datapoint_id__year_EC = None)).select_related("for_datapoint", "for_indicator").values(
            'for_indicator__type_of',
            'value',
            'for_indicator_id',
            'for_datapoint_id__year_EC',
            'for_datapoint_id__year_GC',
            'for_quarter_id',
            'for_month_id__month_AMH',
            'for_quarter_id__title_ENG',
            
        ))
            
            categories_lists_id = queryset.values_list('id', flat=True)

            categories_lists = Category.objects.filter(id__in = categories_lists_id ).values(
                'id',
                'name_ENG',
                'name_AMH', 
            )

        
        paginator = Paginator(queryset, 20) 
        page_number = request.GET.get('page')
        try:
            page_obj = paginator.page(page_number)
        except PageNotAnInteger:
            # if page is not an integer, deliver the first page
            page_obj = paginator.page(1)
        except EmptyPage:
            # if the page is out of range, deliver the last page
            page_obj = paginator.page(paginator.num_pages)

    
        return JsonResponse(
            {
            'categories':list(queryset), 
            'has_previous' : page_obj.has_previous(),
            'has_next' : page_obj.has_next(),
            'previous_page_number' : page_obj.has_previous() and page_obj.previous_page_number() or None,
            'next_page_number' : page_obj.has_next() and page_obj.next_page_number() or None,
            'number' : int(page_obj.number),
            'page_range':list(page_obj.paginator.page_range),
            'num_pages' : page_obj.paginator.num_pages,
            'values':value_filter , 
            'categories_lists': list(categories_lists),
             })


@login_required(login_url='dashboard-login')
@dashboard_user_required
@api_view(['GET'])
def category_detail_lists(request , id):

    if request.method == 'GET':
        category = Category.objects.filter(id = id).first()
        indicators = Indicator.objects.filter(for_category__id = category.pk).select_related()
        
        indicator_list_id = list(indicators.select_related().values_list('id', flat=True))
        month = list(Month.objects.all().values())
        quarter = list(Quarter.objects.all().values())
        value_filter = list(DataValue.objects.filter( Q(for_indicator__id__in=indicator_list_id) & ~Q(for_datapoint_id__year_EC = None)).select_related("for_datapoint", "for_indicator").values(
            'for_indicator__type_of',
            'value',
            'for_indicator_id',
            'for_datapoint_id__year_EC',
            'for_quarter_id',
            'for_month_id__month_AMH',
            'for_month_id__number',
            'for_quarter__title_ENG',
            'for_quarter__number',
        ))

        year = set(DataValue.objects.filter( Q(for_indicator__id__in=indicator_list_id) & ~Q(for_datapoint_id__year_EC = None)).select_related("for_datapoint", "for_indicator").values_list(
            'for_datapoint_id__year_EC',flat=True
        ))


        serialized_indicator = list(indicators.values('id', 'title_ENG', 'type_of'))
        return JsonResponse({'indicators': serialized_indicator,'months' : quarter,'quarters' : month,'values': value_filter, 'year' : list(year)})


@login_required(login_url='dashboard-login')
@dashboard_user_required    
@api_view(['GET'])
def indicator_detail(request, id):
     if request.method == 'GET':
        single_indicator = Indicator.objects.get(pk = id)
        
        indicator_list_id = []
        indicator_list_id.append(single_indicator.pk)

        returned_json = []
        returned_json.append(model_to_dict(single_indicator))

        def child_list(parent):
            for i in list(Indicator.objects.all().values()):
                if i['parent_id'] == parent.id:
                    indicator_list_id.append(i['id'])
                    returned_json.append(i)
                    child_list(Indicator.objects.get(id = i['id']))
                    
        child_list(single_indicator)

        value_filter = list(DataValue.objects.filter( Q(for_indicator__id__in=indicator_list_id) & ~Q(for_datapoint_id__year_EC = None)).select_related("for_datapoint", "for_indicator").values(
            'for_indicator__type_of',
            'value',
            'for_indicator_id',
            'for_datapoint_id__year_EC',
            'for_quarter_id',
            'for_month_id__month_AMH',
        ))

        
        return JsonResponse({'indicators': list(returned_json), 'values' : value_filter})
     
     else:
          return JsonResponse({'indicators': 'failed to access.'})


@login_required(login_url='dashboard-login')
@dashboard_user_required
def search_category_indicator(request):
    queryset = []
    if 'search' in request.GET:
            q = request.GET['search']
            dashboard_topic = DashboardTopic.objects.all()
            queryset = Category.objects.filter().prefetch_related('indicator__set').filter(Q(indicator__title_ENG__contains=q, indicator__for_category__dashboard_topic__in = dashboard_topic ) | Q(indicator__for_category__name_ENG__contains=q, indicator__for_category__dashboard_topic__in = dashboard_topic) ).values(
                'name_ENG',
                'indicator__title_ENG',
            )
    return JsonResponse({'indicators': list(queryset)})
    
        
 
##########################################
#            Chatbot                     #
##########################################
import json
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import WebsiteInfo
from django.shortcuts import render, redirect

def chatbot(request):
    return render(request, "mychat/chatbot.html", {})


from DashboardAPI.forms import WebsiteInfoForm
from django.contrib import messages

def website_info_view(request):
    if request.method == 'POST':
        form = WebsiteInfoForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Website information has been successfully saved.')
            return redirect('website_info')
        else:
            messages.error(request, 'There was an error saving the website information. Please correct the errors below.')
    else:
        form = WebsiteInfoForm()

    website_info = WebsiteInfo.objects.all()
    return render(request, 'user-admin/website_info.html', {'website_info': website_info, 'form': form})

import json
import os
import csv
from django.conf import settings


# Path to the learning data file
LEARNING_DB_PATH = 'learning_db.json'

def learning_data_view(request):
    """View to display the contents of learning_db.json."""
    if os.path.exists(LEARNING_DB_PATH):
        with open(LEARNING_DB_PATH, 'r') as f:
            learning_data = json.load(f)
    else:
        learning_data = {}

    return render(request, 'mychat/learning_data.html', {'learning_data': learning_data})

def load_learning_data():
    """Load learning data from the JSON file."""
    if os.path.exists(LEARNING_DB_PATH):
        with open(LEARNING_DB_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_learning_data(data):
    """Save learning data to the JSON file."""
    with open(LEARNING_DB_PATH, 'w') as f:
        json.dump(data, f, indent=4)

def store_learning_data(query, response):
    """
    Store user query and response in the learning database.
    For production, consider using a database instead of in-memory storage.
    """
    learning_data = load_learning_data()
    
    # Check if the query already exists
    if query.lower() not in learning_data:
        learning_data[query.lower()] = response
        save_learning_data(learning_data)


import re


def csv_to_json(csv_file):
    """Convert CSV file to JSON format."""
    csv_data = csv_file.read().decode('utf-8').splitlines()
    csv_reader = csv.DictReader(csv_data)
    data_list = [row for row in csv_reader]
    return data_list

def extract_keywords(query):
    """Extract key terms from the user query to improve matching."""
    stop_words = ['what', 'is', 'the', 'of', 'value', 'for', 'please', 'provide', 'me', 'with', 'more', 'context', 'information', 'about']
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stop_words]
    return keywords

def find_best_match(keywords, data):
    """Find the best match for the extracted keywords in the data."""
    for keyword in keywords:
        for entry in data:
            if keyword in entry.get("Name", "").lower():
                return entry
    return None

def format_data_for_display(entry):
    """Format a single entry's data into a readable format."""
    if entry:
        name = entry.get("Name", "")
        result = f"Values for {name}:\n"
        for key, value in entry.items():
            if key != "Name" and value != "-":
                result += f"  {key}: {value}\n"
        return result.strip()
    return None

def get_data_from_json(query, json_file_path):
    """Retrieve and format data from the JSON file based on the query."""
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                keywords = extract_keywords(query)
                matched_entry = find_best_match(keywords, data)
                return matched_entry
        except Exception as e:
            return f"Failed to read JSON file: {str(e)}"
    return None

def get_data_from_website_info(query):
    """Retrieve data from WebsiteInfo model based on the query."""
    try:
        website_info = WebsiteInfo.objects.all()
        info_dict = {info.key.lower(): info.value for info in website_info}

        keywords = extract_keywords(query)
        
        for keyword in keywords:
            for key in info_dict:
                if keyword in key:
                    return {key: info_dict[key]}
    except Exception as e:
        return f"Failed to retrieve data from WebsiteInfo: {str(e)}"
    
    return None

def query_gemini_api(text):
    """Query Gemini API to format text into a user-friendly statement."""
    try:
        api_key = 'AIzaSyAZGQ-XFZhLXnVBZ-72jH1hD8laIgM2j_c'  # Replace with your actual API key
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
        headers = {'Content-Type': 'application/json'}
        data_to_send = {
            "contents": [{"role": "user", "parts": [{"text": text}]}]
        }
        response = requests.post(url, headers=headers, json=data_to_send)
        response.raise_for_status()  # Raise an exception for HTTP errors

        response_data = response.json()
        if response_data.get('candidates'):
            return response_data['candidates'][0]['content']['parts'][0]['text']
        else:
            return "No additional content available."
    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)}"

def chat_view(request):
    if request.method == 'POST':
        user_query = request.POST.get('message', '').lower()
        json_file_path = os.path.join(settings.MEDIA_ROOT, 'data.json')

        # Handle CSV file upload
        if 'file' in request.FILES:
            csv_file = request.FILES['file']
            if csv_file.name.endswith('.csv'):
                try:
                    data_list = csv_to_json(csv_file)

                    with open(json_file_path, 'w') as json_file:
                        json.dump(data_list, json_file, indent=4)

                    response = {'message': 'CSV file uploaded and JSON file updated successfully.'}
                    return JsonResponse(response)
                except Exception as e:
                    return JsonResponse({'error': f'Failed to process CSV file: {str(e)}'}, status=500)

        # Handle chat messages
        matched_entry = get_data_from_json(user_query, json_file_path)

        if not matched_entry:
            matched_entry = get_data_from_website_info(user_query)

        if matched_entry:
            formatted_data = format_data_for_display(matched_entry)
            gemini_input_text = f"User query: {user_query}\nMatched data:\n{formatted_data}"
        else:
            gemini_input_text = f"User query: {user_query}\nNo matching data found."

        gemini_response_text = query_gemini_api(gemini_input_text)

        response_text = gemini_response_text or "Sorry, I couldn’t find any information."

        return JsonResponse({'response': response_text})

    return JsonResponse({'error': 'Invalid request method'}, status=400)


from django.core.paginator import Paginator
from .models import WebsiteInfo

def website_trained_list(request):
    website_info_list = WebsiteInfo.objects.all().order_by('key')
    paginator = Paginator(website_info_list, 3)  # Show 5 items per page.

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'user-admin/website_info_list.html', {'page_obj': page_obj})