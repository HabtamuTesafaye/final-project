
from django.shortcuts import get_object_or_404, render, redirect
from django.http import Http404, JsonResponse,HttpResponseRedirect
from django.contrib import messages
from TimeSeriesBase.models import *
from .forms import *
from django.forms.models import model_to_dict
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from UserManagement.decorators import *
from auditlog.models import LogEntry
from datetime import datetime
from TimeSeriesBase.admin import *
from django.http import JsonResponse
from django.db.models import Max
from django.db.models import Count,Prefetch
import random
import json as toJSON
from django.contrib.auth.models import AnonymousUser
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.conf import settings
from django.shortcuts import render
import os


def site_configuration_view(request):
    # Fetch the first instance of SiteConfiguration from the database
    site_config = SiteConfiguration.objects.first()

    if request.method == 'POST':
        # If the form is submitted via POST, populate it with the request data and the instance
        form = SiteConfigurationForm(request.POST, instance=site_config)
        if form.is_valid():
            # If the form is valid, save the changes
            form.save()
            messages.success(request, 'Site configuration updated successfully.')
            return redirect('user-admin-index')  # Redirect to the specified URL after updating the configuration
    else:
        # If the request method is not POST, create a form instance with the fetched site_config
        form = SiteConfigurationForm(instance=site_config)

    # Render the HTML template with the form
    return render(request, 'user-admin/index.html', {'form': form})


##############################
#          JSON             #
#############################
##LIST VIEW START
from django.core.cache import cache
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


@login_required(login_url='login')
@admin_user_required
def filter_category_lists(request,pk):
    topic = Topic.objects.get(pk = pk)
    category_lists = list(Category.objects.filter(topic = topic).prefetch_related('topic').values())
    return JsonResponse(category_lists, safe=False)


@login_required(login_url='login')
@admin_user_required
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
   

@login_required(login_url='login')
@admin_user_required
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

##LIST VIEW END


#Indicator Page 
@login_required(login_url='login')
@admin_user_required
def filter_indicator_json(request):
    topic = list(Topic.objects.all().values())
    category_data = list(Category.objects.all().values())
    indicator = list(Indicator.objects.filter(~Q(for_category_id = None )).values())
    measurements = list(Measurement.objects.filter().values())
    context = {
        'topics' : topic,
        'categories' : category_data,
        'indicators' : indicator,
        'measurements' : measurements
    }

    return JsonResponse(context)


#Indicator Page Detail Indicator 
@login_required(login_url='login')
@admin_user_required
def filter_indicator_indicator_page(request, pk):
    single_indicator = Indicator.objects.get(pk=pk)
    returned_json = []
    returned_json.append(model_to_dict(single_indicator))
    indicators = list(Indicator.objects.select_related('parent').filter().values())
    indicator_point = list(Indicator_Point.objects.filter(for_indicator=pk).values())

    def child_list(parent):
        for i in indicators:
            if i['parent_id'] == parent['id']:  # Accessing 'parent_id' instead of 'parent.id'
                returned_json.append(i)
                child_list(i)  # Calling child_list with the child indicator as the parent

    child_list(model_to_dict(single_indicator))  # Passing the dictionary representation of the single_indicator

    context = {
        'indicators': returned_json,
        'indicator_point': indicator_point,
    }

    return JsonResponse(context)


#Indicator Detail Page With Child and with Values
@login_required(login_url='login')
@admin_user_required
def filter_indicator(request, pk):
    single_indicator = Indicator.objects.get(pk=pk)
    returned_json = []
    year = list(DataPoint.objects.all().values())
    returned_json.append(model_to_dict(single_indicator))
    indicators = list(Indicator.objects.all().values())
    indicator_point = list(Indicator_Point.objects.filter(for_indicator=pk).values())
    measurements = list(Measurement.objects.all().values())

    # Initialize month_data and quarter_data before checking if they are None
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
    indicator_list += [model_to_dict(child_indicator) for child_indicator in indicators_with_children]




    def child_list(parent):
        for i in indicators:
            if i['parent_id'] == parent.id:
                returned_json.append(i)
                child_list(Indicator.objects.get(id=i['id']))

    child_list(single_indicator)

    value_new = []
    year_new = []

    # Fetch data values for each indicator in a single query
    for indicator in returned_json:
        # Fetch DataValues and related DataPoint instances in a single query
        value_filter = DataValue.objects.filter(for_indicator__id=indicator['id']).select_related(
            "for_datapoint", "for_indicator")

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
        'indicators': returned_json,
        'indicator_point': indicator_point,
        'year': year,
        'new_year': year_new,
        'value': value_new,
        'measurements': measurements,
        'month': month_data,
        'quarter': quarter_data
    }
    return JsonResponse(context)



#Source Page
@login_required(login_url='login')
def json_filter_source(request):
    sources = Source.objects.all()
    sources_data = []
    for source in sources:
        sources_data.append({
            'id': source.id,
            'title_ENG': source.title_ENG,
            'title_AMH': source.title_AMH,
            'updated': source.updated.isoformat(),
            'created': source.created.isoformat(),
            'is_deleted': source.is_deleted,
        })

    return JsonResponse({'sources': sources_data}) 

#Category Page -> For Edit
@login_required(login_url='login')
def filter_category_json(request):
    category_data = list(Category.objects.filter().select_related('topic').values('id', 'name_ENG', 'name_AMH','topic_id','topic__title_ENG','topic__title_AMH'))
    context = {
        'categories': category_data,
    }

    return JsonResponse(context)

#MEASUREMENT PAGE
@login_required(login_url='login')
def json_measurement(request):
    measurements = list(Measurement.objects.all().values())
    
    context = {
        'measurements' : measurements
    }
    return JsonResponse(context)


#TOPIC PAGE
@login_required(login_url='login')
def json_filter_topic(request):
    topics = Topic.objects.all()
    
    # Creating a list of dictionaries representing each topic
    topics_data = []
    for topic in topics:
        topics_data.append({
            'id': topic.id,
            'title_ENG': topic.title_ENG,
            'title_AMH': topic.title_AMH,
            'updated': topic.updated.isoformat(),
            'created': topic.created.isoformat(),
            'is_deleted': topic.is_deleted,
        })

    # Returning the list as JSON
    return JsonResponse({'topics': topics_data})


@login_required(login_url='login')
@admin_user_required
def dashboard_json(request):
    topic = list(Topic.objects.all().values())
    category = list(Category.objects.filter().select_related('topic').values())
    indicator = list(Indicator.objects.filter().select_related('for_category').values())
    indicator_point = list(Indicator_Point.objects.filter().select_related('for_indicator').values())
    year =list( DataPoint.objects.all().values())
    month = list(Month.objects.all().values())
    quarter = list(Quarter.objects.all().values())
    measurement = list(Measurement.objects.all().values())
    values = list(DataValue.objects.all().select_related("for_datapoint", "for_indicator").values())
    indicator_data = indicator



    context = {
        'topics': topic,
        'categories': category,
        'indicators':indicator_data,
        'indicator_point' : indicator_point,
        'year' : year,
        'quarter' : quarter,
        'month' : month,
        'value' : values

    }
    return JsonResponse(context)


@login_required(login_url='login')
@admin_user_required
def json_random(request):
    # Fetch all categories
    categories = Category.objects.all()

    # Choose a random category
    random_category = None
    while not random_category:
        random_category = random.choice(categories)
        
        # Fetch parent indicators for the selected category
        parent_indicators = Indicator.objects.filter(for_category=random_category, parent=None)

        # Fetch year data for the selected category and parent indicators
        year_data = DataValue.objects.filter(for_indicator__in=parent_indicators, is_deleted=False).exclude(for_datapoint__year_EC__isnull=True).values('for_indicator__title_ENG', 'for_datapoint__year_EC', 'value')

        # Check if there are any parent indicators with associated non-null year and value data
        if not any(year_data):
            random_category = None

    # Convert data to a dictionary with indicator names as keys
    indicators_data = {ind['for_indicator__title_ENG']: [] for ind in year_data}

    # Populate the dictionary with year and value data
    for data_point in year_data:
        # Check if the year is not null before adding it to the response
        if data_point['for_datapoint__year_EC'] is not None:
            indicators_data[data_point['for_indicator__title_ENG']].append({
                'year': data_point['for_datapoint__year_EC'],
                'value': float(data_point['value'])
            })
    
    # Directly return the indicators_data dictionary
    return JsonResponse(indicators_data)

@login_required(login_url='login')
@admin_user_required
def json_filter_year(request):
    try:
        # Determine the largest year from the DataPoint model
        largest_year_instance = DataPoint.objects.latest('year_EC')
        largest_year = int(largest_year_instance.year_EC)

        # Calculate the next year
        new_year = largest_year + 1

        # Create a new DataPoint instance with the next year
        DataPoint.objects.create(year_EC=str(new_year))
        
        # Adding a message for success
        messages.success(request, 'Year added successfully!')

        return JsonResponse({'success': True, 'new_year': new_year})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

# calculate for drildown to avoid nested loop 
def calculate_category_data(topic, categories_with_indicators):
    return [
        [
            category.name_ENG,
            len(category.indicators)
        ] for category in categories_with_indicators if category.topic == topic
    ]

@login_required(login_url='login')
@admin_user_required   
def json_filter_drilldown(request):
    # Fetch topics with annotated category count
    topics = Topic.objects.annotate(category_count=Count('category'))

    # Fetch categories and related indicators for all topics using prefetch_related
    categories_with_indicators = Category.objects.filter(topic__in=topics).prefetch_related(
        Prefetch('indicator_set', queryset=Indicator.objects.filter(children__isnull=True), to_attr='indicators')
    )

    # Create topic_data
    topic_data = {
        "name": "Topic",
        "colorByPoint": True,
        "data": [
            {
                "name": topic.title_ENG,
                "y": topic.category_count,
                "drilldown": topic.title_ENG
            } for topic in topics
        ]
    }

    # Create drilldown data
    drilldown = [
        {
            "name": topic.title_ENG,
            "id": topic.title_ENG,
            "data": calculate_category_data(topic, categories_with_indicators)
        } for topic in topics
    ]

    return JsonResponse({
        "topic_data": topic_data,
        "drilldown": drilldown
    })


##############################
#          Audit            #
#############################

@login_required(login_url='login')
@admin_user_required
def audit_log_list(request):
    auditlog_entries = LogEntry.objects.select_related('content_type', 'actor')[:1500]
    return render(request, 'user-admin/audit.html', {'auditlog_entries': auditlog_entries})


##############################
#          INDEX             #
#############################

from django.db.models import Count

def index(request):
    auditlog_entries = LogEntry.objects.select_related('content_type', 'actor')[:6]
    size_topic = Topic.objects.filter(is_deleted=False).aggregate(count=Count('id'))['count']
    size_category = Category.objects.filter(is_deleted=False).aggregate(count=Count('id'))['count']
    size_indicator = Indicator.objects.filter(is_deleted=False).aggregate(count=Count('id'))['count']
    size_source = Source.objects.filter(is_deleted=False).aggregate(count=Count('id'))['count']


    context = {
        'size_topic': size_topic,
        'size_category': size_category,
        'size_indicator': size_indicator,
        'size_source': size_source,
        'auditlog_entries': auditlog_entries,
    }

    return render(request, 'user-admin/index.html', context)

 
##############################
#          Category          #
#############################
@login_required(login_url='login')
@admin_user_required
def category(request, category_id=None):
    categories = Category.objects.filter().select_related('topic')
    form_file = ImportFileForm()

    form = catagoryForm()
    global imported_data_global

    if request.method == 'POST':
        
        category_id_str = request.POST.get('catagory_Id')
        try: 
            category_obj = Category.objects.get(pk = category_id_str)
            form = catagoryForm(request.POST, instance = category_obj)
            form_instance = True
        except: 
            category_obj = None
            form = catagoryForm(request.POST )
            form_instance = False
            
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Updated!') if form_instance else messages.success(request, 'Successfully Added!')


        elif 'fileCategoryValue' in request.POST:
            form_file = ImportFileForm(request.POST, request.FILES)
            if form_file.is_valid():
                file = request.FILES['file']
                success, imported_data, result = handle_uploaded_Category_file(file)
                imported_data_global = imported_data
                context = {'result': result}
                return render(request, 'user-admin/import_preview.html', context=context)
            else:
                messages.error(request, 'File not recognized')

        elif 'confirm_data_form' in request.POST:
            success, message = confirm_file(imported_data_global, 'category')
            if success:
                messages.success(request, message)
            else:
                messages.error(request, message)

    context = {
        'form': form,
        'categories': categories,
        'formFile': form_file
    }
    return render(request, 'user-admin/categories.html', context=context)


@login_required(login_url='login')
@admin_user_required
def delete_category(request, pk):
    category = Category.objects.get(pk=pk)
    previous_page = request.META.get('HTTP_REFERER')
    
    # Soft delete the category
    category.is_deleted = True
    category.save()

    # Optionally, you can soft delete related objects here if needed
    
    messages.success(request, "Successfully Deleted!")
    return HttpResponseRedirect(previous_page)



##############################
#          LIST VIEW         #
#############################
@login_required(login_url='login')
@admin_user_required
def data_list(request):
    global imported_data_global
    formFile = ImportFileIndicatorAddValueForm()
    if request.method == 'POST':
        formFile = ImportFileIndicatorAddValueForm(request.POST, request.FILES )
        if formFile.is_valid():
                file = request.FILES['file']
                type_of_data = formFile.cleaned_data['type_of_data']
                success, imported_data,result = handle_uploaded_DataValue_file(file, type_of_data)
                imported_data_global = imported_data
                context = {
                        'result' : result
                        }
                formFile = ImportFileIndicatorAddValueForm()
                return render(request, 'user-admin/import_preview.html', context=context)
    
        elif 'confirm_data_form' in request.POST:
            success, message = confirm_file(imported_data_global, 'data_value')
            if success:
                formFile = ImportFileIndicatorAddValueForm()
                messages.success(request, message)
            else:
                formFile = ImportFileIndicatorAddValueForm()
                messages.error(request,message)

            
    context = {
        'formFile' : formFile
    }
    return render(request, 'user-admin/data_list_view.html', context)


@login_required(login_url='login')
@admin_user_required
def data_list_detail(request, pk):
    form = ValueForm()
    form_update = ValueForm2()
    sub_indicator_form = SubIndicatorFormDetail()
    indicator = Indicator.objects.get(pk = pk)
    measurement_form = MeasurementSelectForm()
    operation = operationForm()

    if request.method == 'POST':
        if 'addValueIndicator' in request.POST:
            form = ValueForm(request.POST)
            if form.is_valid():
                if indicator.type_of == 'yearly':
                    try:
                        indicator_id = request.POST.get('indicator') 
                        data_point_id = request.POST.get('data_point')
                        value = form.cleaned_data['value']
                        indicator_obj = Indicator.objects.get(pk = indicator_id)
                        data_point_obj = DataPoint.objects.get(pk = data_point_id)
                    
                        value_obj = DataValue()
                        value_obj.value = value
                        value_obj.for_datapoint = data_point_obj
                        value_obj.for_indicator = indicator_obj
                        value_obj.save()
                        form = ValueForm()
                        messages.success(request, 'Successfully Added!')
                        return redirect(request.path)
                    except: 
                        messages.error(request, 'Please Try Again To Edit Indicator!')
                elif indicator.type_of == 'monthly':
                    try:
                        indicator_id = request.POST.get('indicator') 
                        data_point_id = request.POST.get('data_point')
                        month_id = request.POST.get('month')
                        value = form.cleaned_data['value']


                        indicator_obj = Indicator.objects.get(pk = indicator_id)
                        data_point_obj = DataPoint.objects.get(pk = data_point_id)
                        month_obj = Month.objects.get(pk = month_id)
                    
                        value_obj = DataValue()
                        value_obj.value = value
                        value_obj.for_datapoint = data_point_obj
                        value_obj.for_indicator = indicator_obj
                        value_obj.for_month = month_obj
                        value_obj.save()
                        form = ValueForm()
                        messages.success(request, 'Successfully Added!')
                        return redirect(request.path)
                    except: 
                        messages.error(request, 'Please Try Again To Edit Indicator!')              
                elif indicator.type_of == 'quarterly':
                    try:
                        indicator_id = request.POST.get('indicator') 
                        data_point_id = request.POST.get('data_point')
                        quarter_id = request.POST.get('quarter')
                        value = form.cleaned_data['value']


                        indicator_obj = Indicator.objects.get(pk = indicator_id)
                        data_point_obj = DataPoint.objects.get(pk = data_point_id)
                        quarter = Quarter.objects.get(pk = quarter_id)
                    
                        value_obj = DataValue()
                        value_obj.value = value
                        value_obj.for_datapoint = data_point_obj
                        value_obj.for_indicator = indicator_obj
                        value_obj.for_quarter = quarter
                        value_obj.save()
                        form = ValueForm()
                        messages.success(request, 'Successfully Added!')
                        return redirect(request.path)
                    except: 
                        messages.error(request, 'Please Try Again To Edit Indicator!')
        
        if 'editFormIndicatorValue' in request.POST:
            form_update = ValueForm2(request.POST)
            if form_update.is_valid():
                try:  
                    value = form_update.cleaned_data['value2']
                    value_id = request.POST.get('data_value')
                    data_value = DataValue.objects.get(pk = value_id)
                    data_value.value = value
                    data_value.save()
                    form_update = ValueForm2()
                    messages.success(request, 'Successfully Added!')
                    return redirect(request.path)
                except:
                    messages.error(request, 'Please Try Again To Update Indicator!')
        
        if 'formAddIndicator' in request.POST:
            sub_indicator_form = SubIndicatorFormDetail(request.POST)
            if sub_indicator_form.is_valid():
                try: 
                    indicator_id = request.POST.get('addNewIndicator')
                    indicator = Indicator.objects.get(pk = indicator_id)
                    new_sub_indicator = Indicator()
                    new_sub_indicator.title_ENG = sub_indicator_form.cleaned_data['title_ENG']
                    new_sub_indicator.title_AMH =  sub_indicator_form.cleaned_data['title_AMH']
                    new_sub_indicator.parent =  indicator
                    new_sub_indicator.save()
        
                    sub_indicator_form = SubIndicatorForm()
                    messages.success(request, 'Successfully Added!')
                    return redirect(request.path)
                except: 
                    messages.error(request, 'Please Try Again To Add New Sub-Indicator!')
        
        if 'measurementFormId' in request.POST:
            measurement_form = MeasurementSelectForm(request.POST)
            if measurement_form.is_valid():
                try:
                    measurement_id = measurement_form.cleaned_data['measurement_form']
                    measurement = Measurement.objects.get(pk = measurement_id)
                    indicator.measurement = measurement
                    indicator.save()
                    messages.success(request, 'Successfully measurement Updated!')
                    return redirect(request.path)
                except:
                    messages.error(request, 'Please Try Again!')
            else:
                messages.error(request, 'Please Try Again not valid!')
        
        if 'indicatorYearId' in request.POST:
            is_actual = request.POST.get('isActualInput')
            is_actual_data_point_id = request.POST.get('indicatorYearId')

            try:    
                indicator_point = Indicator_Point.objects.get(for_indicator = indicator, for_datapoint = is_actual_data_point_id)
            except:
                indicator_point = None

            try:
                data_point = DataPoint.objects.get(pk = is_actual_data_point_id)
            except:
                data_point = None

            #Is Indicator Point is Found
            if(indicator_point):
                if(is_actual):
                    indicator_point.is_actual = True
                else:
                    indicator_point.is_actual = False
                 
                indicator_point.save()
                messages.success(request, 'Successfully Actual Point updated!')
            elif(data_point):
                indicator_obj = Indicator_Point()
                indicator_obj.for_indicator = indicator
                indicator_obj.for_datapoint = data_point
                if(is_actual):
                    indicator_obj.is_actual = True
                else:
                    indicator_obj.is_actual = False
                
                indicator_obj.save()
                messages.success(request, 'Successfully Actual Point Added!')
            else:
                messages.error(request, 'Please Try Again!')
        
        if 'editOperation' in request.POST:
            operation = operationForm(request.POST)
            if operation.is_valid():
                try:
                   indicator_id = request.POST.get('indicator_operator')
                   op = operation.cleaned_data['operation_type']
                   indicator_obj = Indicator.objects.get(pk = indicator_id)
                   indicator_obj.op_type = op
                   indicator_obj.save()
                   messages.success(request, 'Successfully Operator Updated!')
                   return redirect(request.path)
                except:
                   messages.error(request, 'Please Try Again not valid!')
                   

                
            
                       
    context = {
        'form' : form,
        'form_update' : form_update,
        'sub_indicator_form' : sub_indicator_form,
        'indicator' : indicator,
        'measurement_form' :measurement_form,
        'operationForm' : operation
    }
    return render(request, 'user-admin/data_list_detail.html', context)


##############################
#          INDICATOR         #
#############################
#Indicator 
@login_required(login_url='login')
@admin_user_required
def indicator(request):
    add_indicator = IndicatorForm(request.POST or None)
    formFile = ImportFileIndicatorForm()
    global imported_data_global 
    if request.method == 'POST':
        if 'formAddIndicator' in request.POST:
            if add_indicator.is_valid():
                add_indicator.save()
                messages.success(request, 'Successfully Added!')
            else:
                messages.error(request, 'Please Try Again!')

        if 'fileIndicatorFile' in request.POST:
            formFile = ImportFileIndicatorForm(request.POST, request.FILES )
            if formFile.is_valid():
                file = request.FILES['file']
                category = formFile.cleaned_data['category']
                success, imported_data,result =  handle_uploaded_Indicator_file(file, category)
                imported_data_global = imported_data
                context = {
                        'result' : result
                        }
                return render(request, 'user-admin/import_preview.html', context=context)
            else:
                messages.error(request, 'File not recognized')
        elif 'confirm_data_form' in request.POST:
            success, message = confirm_file(imported_data_global, 'indicator')
            if success:
                messages.success(request, message)
            else:
                messages.error(request,message)
    
    context = {
        'add_indicator' : add_indicator,
        'formFile' : formFile
    }
    return render(request, 'user-admin/indicators.html', context)

@login_required(login_url='login')
@admin_user_required
def indicator_list(request, pk):
    add_indicator = IndicatorForm()
    category = Category.objects.get(pk = pk)
    indicator_list = Indicator.objects.filter(for_category = category)
    form = IndicatorForm()
    formFile = ImportFileIndicatorForm()
    if request.method == "POST":
        if 'form_indicator_edit_dynamic' in request.POST:
            form = IndicatorForm(request.POST)
            if form.is_valid():
                title_ENG = form.cleaned_data['title_ENG']
                title_AMH = form.cleaned_data['title_AMH']
                category_obj = form.cleaned_data['for_category']
                type_of_obj = form.cleaned_data['type_of']
                indicator_id = request.POST.get('indicator_Id')
                measurement = request.POST.get('measurement_form')
                operation_type = form.cleaned_data['operation_type']
                is_public = form.cleaned_data['is_public']
                measurement_obj = Measurement.objects.get(pk = measurement)
                Indicator.objects.filter(id = indicator_id).update(title_AMH = title_AMH,title_ENG = title_ENG,for_category = category_obj, type_of = type_of_obj, op_type = operation_type, measurement = measurement_obj, is_public = is_public)
                form = IndicatorForm()
                messages.success(request, 'Successfully Updated')
            else:
                error_messages = ''
                messages.error(request, f'Please Try Again, {error_messages}')

        if 'formAddIndicator' in request.POST:
            add_indicator = IndicatorForm(request.POST)
            if add_indicator.is_valid():
                title_ENG = add_indicator.cleaned_data['title_ENG']
                title_AMH = add_indicator.cleaned_data['title_AMH']
                category_obj = add_indicator.cleaned_data['for_category']
                type_of_obj = add_indicator.cleaned_data['type_of']
                operation_type = add_indicator.cleaned_data['operation_type']
                is_public = add_indicator.cleaned_data['is_public']

                obj = Indicator()
                obj.title_AMH = title_AMH
                obj.title_ENG = title_ENG
                obj.for_category = category_obj
                obj.type_of = type_of_obj
                obj.op_type = operation_type
                obj.is_public = is_public
                
                try:
                    obj.save()
                    add_indicator = IndicatorForm()
                    messages.success(request, 'Successfully Added!')
                except:
                    messages.error(request, 'Please Try again! Indicator Exist.')
            else:
                messages.error(request, 'Please Try again! or May the Indicator Exist.')
        
        if 'fileIndicatorFile' in request.POST:
            formFile = ImportFileIndicatorForm(request.POST, request.FILES )
            if formFile.is_valid():
                file = request.FILES['file']
                category = formFile.cleaned_data['category']
                success, message = handle_uploaded_Indicator_file(file, category)
                
                if success:
                    messages.success(request, message)
                else:
                    messages.error(request, message)
    
            else:
                messages.error(request, 'File not recognized')

    context = {
        'indicators' : indicator_list,
        'category' : category,
        'form' : form,
        'add_indicator' : add_indicator,
        'formFile' : formFile,
        
    }
    return render(request, 'user-admin/indicators.html', context)

@login_required(login_url='login')
@admin_user_required
def indicator_detail(request, pk):
    indicator = Indicator.objects.get(pk = pk)
    indicator_list = Indicator.objects.filter(for_category = indicator.for_category)
    editIndicator = IndicatorSubForm()
    addIndicator  = SubIndicatorForm()
    operation = operationForm()

    if request.method == 'POST':
        if 'editSubIndicatorForm' in request.POST:
            editIndicator = IndicatorSubForm(request.POST)
            if editIndicator.is_valid():
                indicator_id = request.POST.get('indicator_Id')
                indicator_title_AMH = editIndicator.cleaned_data['title_AMH']
                indicator_title_ENG = editIndicator.cleaned_data['title_ENG']
                operation_type = editIndicator.cleaned_data['operation_type']
                is_public = editIndicator.cleaned_data['is_public']
                
                try:
                    indicator_obj  = Indicator.objects.get(pk = indicator_id)
                    indicator_obj.title_AMH = indicator_title_AMH.strip()
                    indicator_obj.title_ENG = indicator_title_ENG.strip()
                    indicator_obj.op_type = operation_type
                    indicator_obj.is_public = is_public
                    indicator_obj.save()
                    messages.success(request, 'Successfully Updated!')
                except:
                    messages.error(request, 'Please Try Again!')
        if 'addSubIndicatorForm' in request.POST:
            addIndicator  = SubIndicatorForm(request.POST)
            if addIndicator.is_valid():
                parent_id = request.POST.get('addNewIndicator')
                indicator_title_AMH = addIndicator.cleaned_data['title_AMH_add']
                indicator_title_ENG = addIndicator.cleaned_data['title_ENG_add']
                is_public = addIndicator.cleaned_data['is_public']
                try:
                    parent_obj = Indicator.objects.get(pk = parent_id)
                    new_indicator = Indicator()
                    new_indicator.title_AMH = indicator_title_AMH
                    new_indicator.title_ENG = indicator_title_ENG
                    new_indicator.parent = parent_obj
                    new_indicator.is_public = is_public
                    new_indicator.save()
                    messages.success(request, 'Successfully Added!')
                except:
                    messages.error(request, 'Please Try Again!')
       
        


    context = {
        'indicators' : indicator_list,
        'category' : category,
        'editIndicator' : editIndicator,
        'indicator' : indicator,
        'addIndicator' : addIndicator,
    }
    return render(request, 'user-admin/indicator_detail.html', context)

@login_required(login_url='login')
@admin_user_required
def delete_indicator(request,pk):
    
    indicator = Indicator.objects.get(pk=pk)
    previous_page = request.META.get('HTTP_REFERER')

    #Parent Indicator 
    indicator.is_deleted = True
    indicator.save()
    
    indicator_list = Indicator.objects.all()

    #Check Child of Child 
    def check_child(parent_obj):
        for indicator_obj in indicator_list:
            if indicator_obj.parent == parent_obj:
                indicator_obj.is_deleted = True
                indicator_obj.save()
                check_child(indicator_obj)


    for indicator_obj in indicator_list:
        if indicator_obj.parent == indicator:
            indicator_obj.is_deleted = True
            indicator_obj.save()
            check_child(indicator_obj)


    #Parent Related Values 
    years = DataPoint.objects.all()
    for year in  years:
        try: 
           deleted_indicator = DataValue.objects.get(for_datapoint = year, for_indicator = indicator)
           deleted_indicator.is_deleted = True
           deleted_indicator.save()
        except:
            None
    messages.success(request, "Successfully Removed!")
    return HttpResponseRedirect(previous_page)

 
##############################
#          MEASUREMENT       #
#############################
       
@login_required(login_url='login')
@admin_user_required
def measurement(request):
    addMeasurementForm = MeasurementForm()
    editMeasurementForm = MeasurementForm()
    addNewMeasurementForm = MeasurementForm()
    formFile = ImportFileForm()
    global imported_data_global 
    if request.method == 'POST':
        if 'formAddMeasurement' in request.POST:
            addMeasurementForm = MeasurementForm(request.POST)
            if addMeasurementForm.is_valid():
                try:
                   new_measurement = addMeasurementForm.save(commit=False)
                   parent_id = request.POST.get('addNewMeasurement')
                   parent = Measurement.objects.get(pk = parent_id)
                   new_measurement.parent = parent
                   new_measurement.save()
                   addMeasurementForm = MeasurementForm()
                   messages.success(request, 'Successfully New measurement Added!')
                except:
                    messages.error(request, 'Please Try Again!')
        if 'form_measurement_edit' in request.POST:
            editMeasurementForm = MeasurementForm(request.POST)
            if editMeasurementForm.is_valid():
                measurement_id = request.POST.get('id_measurement')
                measurement_obj = Measurement.objects.get(pk = measurement_id)
                measurement_obj.Amount_AMH = editMeasurementForm.cleaned_data['Amount_AMH']
                measurement_obj.Amount_ENG = editMeasurementForm.cleaned_data['Amount_ENG']
                measurement_obj.save()
                editMeasurementForm = MeasurementForm()
                messages.success(request, 'Successfully Updated')
            else:
                messages.error(request, 'Please Try Again!')
        
        if 'addNewMeasurementValue' in request.POST:
            addNewMeasurementForm = MeasurementForm(request.POST)
            if addNewMeasurementForm.is_valid():
                addNewMeasurementForm.save()
                messages.success(request, 'Successfully Added!')
            else:
                messages.error(request, 'Please Try Again!')
        
        if 'fileMeasurementFile' in request.POST:
            formFile = ImportFileForm(request.POST, request.FILES )
            if formFile.is_valid():
                file = request.FILES['file']
                success, imported_data,result = handle_uploaded_Measurement_file(file)
                imported_data_global = imported_data
                context = {
                        'result' : result
                        }
                return render(request, 'user-admin/import_preview.html', context=context)
    
            else:
                messages.error(request, 'File not recognized')
        elif 'confirm_data_form' in request.POST:
            success, message = confirm_file(imported_data_global, 'measuremennt')
            if success:
                messages.success(request, message)
            else:
                messages.error(request,message)

                
    context = {
        'addMeasurementForm' : addMeasurementForm,
        'editMeasurementForm' : editMeasurementForm,
        'addNewMeasurementForm' : addNewMeasurementForm,
        'formFile' : formFile
    }
    return render(request, 'user-admin/measurement.html', context)

@login_required(login_url='login')
@admin_user_required
def delete_measurement(request, pk):
    try:
        measurement = Measurement.objects.get(pk = pk)
        previous_page = request.META.get('HTTP_REFERER')
        measurement.is_deleted = True
        measurement.save()
    
        messages.success(request, "Successfully Removed")
        return HttpResponseRedirect(previous_page)
    except: 
        messages.error(request, 'Please Try Again!')


##############################
#          SOURCE            #
#############################
@login_required(login_url='login')
@admin_user_required
def source(request, source_id=None):
    sources = Source.objects.all()

    if request.method == 'POST':
        if 'source_id' in request.POST:
            # Editing an existing source
            source_instance = get_object_or_404(Source, id=request.POST['source_id'])
            form = SourceForm(request.POST, instance=source_instance)
        else:
            # Adding a new source
            form = SourceForm(request.POST)
        
        if form.is_valid():
            form.save()
            if 'source_id' in request.POST:
                messages.success(request, "Source has been successfully updated!")
            else:
                messages.success(request, "Source has been successfully added!")
            return redirect('user-admin-source')
        else:
            messages.error(request, "Value exists or please try again!")
    else:
        # GET request or form is not valid, display the form
        if source_id:
            # Editing an existing source, populate the form with existing data
            source_instance = get_object_or_404(Source, id=source_id)
            form = SourceForm(instance=source_instance)
        else:
            # Adding a new source
            form = SourceForm()

    context = {
        'form': form,
        'sources': sources
    }
    return render(request, 'user-admin/source.html', context=context)

@login_required(login_url='login')
@admin_user_required
def delete_source(request,pk):
    source = Source.objects.get(pk=pk)
    previous_page = request.META.get('HTTP_REFERER')
    
    # Soft delete the category
    source.is_deleted = True
    source.save()

    # Optionally, you can soft delete related objects here if needed
    
    messages.success(request, "Successfully Deleted!")
    return HttpResponseRedirect(previous_page)



##############################
#         TOPIC              #
#############################
@login_required(login_url='login')
@admin_user_required
def topic(request, topic_id=None):
    topics = Topic.objects.filter(is_deleted=False)
    topic_instance = None
    formFile = ImportFileForm()

    if topic_id:
        topic_instance = get_object_or_404(Topic, pk=topic_id)

    form = TopicForm(instance=topic_instance)
    global imported_data_global

    if request.method == 'POST':
        if 'topic_Id' in request.POST or 'topicFormValue' in request.POST:
            topic_id = request.POST.get('topic_Id')
            if topic_id:
                try:
                    topic_instance = get_object_or_404(Topic, id=topic_id)
                    form = TopicForm(request.POST, instance=topic_instance)
                except Http404:
                    form = TopicForm(request.POST)
            else:
                form = TopicForm(request.POST)

            if form.is_valid():
                obj = form.save(commit=False)
                obj.save()

                messages.success(request, "Topic has been successfully added!" if not topic_id else "Topic has been successfully updated!")
                return redirect('user-admin-topic')
            else:
                messages.error(request, "Value exists or please try again!")

        if 'fileTopicValue' in request.POST:
            formFile = ImportFileForm(request.POST, request.FILES)
            if formFile.is_valid():
                file = request.FILES['file']
                success, imported_data, result = handle_uploaded_Topic_file(file)
                imported_data_global = imported_data
                context = {'result': result}
                return render(request, 'user-admin/import_preview.html', context=context)
            else:
                messages.error(request, 'File not recognized')

        elif 'confirm_data_form' in request.POST:
            success, message = confirm_file(imported_data_global, 'topic')
            if success:
                messages.success(request, message)
            else:
                messages.error(request, message)

    context = {
        'form': form,
        'topics': topics,
        'topic_id': topic_id,
        'formFile': formFile
    }
    return render(request, 'user-admin/topic.html', context=context)


@login_required(login_url='login')
@admin_user_required
def delete_topic(request,pk):
    topic = Topic.objects.get(pk=pk)
    previous_page = request.META.get('HTTP_REFERER')
    
    # Soft delete the category
    topic.is_deleted = True
    topic.save()

    # Optionally, you can soft delete related objects here if needed
    
    messages.success(request, "Successfully Deleted!")
    return HttpResponseRedirect(previous_page)
 

##############################
#         TRASH             #
#############################  
@login_required(login_url='login')
@admin_user_required
def trash_topic(request):
    if request.method == 'POST':
        topic_id = request.POST.get('topic_id')
        if topic_id:
            topic = get_object_or_404(Topic, pk=topic_id)
            topic.is_deleted = False
            topic.save()
            messages.success(request, 'Topic restored successfully.')
            return redirect('trash-topic')
        else:
            messages.error(request, 'Failed to restore topic.')

    recycled_topics = Topic.objects.filter(is_deleted=True)
    
    context = {
        'recycled_topics': recycled_topics,
    }
    return render(request, 'user-admin/trash_Topic.html', context)


@login_required(login_url='login')
@admin_user_required
def trash_indicator(request):
    return render(request, 'user-admin/trash_Indicator.html')

@login_required(login_url='login')
@admin_user_required
def trash_category(request):
    if request.method == 'POST':
        catagory_Id = request.POST.get('catagory_Id')
        if catagory_Id:
            category = get_object_or_404(Category, pk=catagory_Id)
            category.is_deleted = False
            category.save()
            messages.success(request, 'Catagory restored successfully.')
            return redirect('trash-category')
        else:
            messages.error(request, 'Failed to restore Catagory.')

    recycled_categories = Category.objects.filter(is_deleted=True)
    context = {
        'recycled_categories': recycled_categories,
    }
    return render(request, 'user-admin/trash_Category.html', context)

@login_required(login_url='login')
@admin_user_required
def trash_source(request):
    if request.method == 'POST':
        source_id = request.POST.get('source_id')
        if source_id:
            source = get_object_or_404(Source, pk=source_id)
            source.is_deleted = False
            source.save()
            messages.success(request, 'Source restored successfully.')
            return redirect('trash-source')
        else:
            messages.error(request, 'Failed to restore Source.')

    recycled_sources = Source.objects.filter(is_deleted=True)
    context = {
        'recycled_sources': recycled_sources,
    }
    return render(request, 'user-admin/trash_Source.html', context)


##############################
#         RESTORE            #
#############################

@login_required(login_url='login')
@admin_user_required
def restore_item(request, item_type, item_id):
    previous_page = request.META.get('HTTP_REFERER')
    model_mapping = {
        'topic': Topic,
        'indicator': Indicator,
        'catagory': Category,
        'source': Source,
    }

    model = model_mapping.get(item_type)
    if not model:  
        messages.error(request,'Failed to restore item')
        return HttpResponseRedirect(previous_page) # Change to the actual view name for recycled items


    item = get_object_or_404(model, pk=item_id)
    item.is_deleted = False
    item.save()

    messages.success(request,'Successfully restored')

    # Redirect to the view where the recycled items are displayed
    return redirect('user-admin-recyclebin') 


@login_required
@admin_user_required
def restore_indicator(request, pk):
    indicator = Indicator.objects.get(pk=pk)
    previous_page = request.META.get('HTTP_REFERER')

    #Parent Indicator 
    indicator.is_deleted = False
    indicator.save()

    indicator_list = Indicator.objects.all()


    #Check Child of Child 
    def check_child(parent_obj):
        for indicator_obj in indicator_list:
            if indicator_obj.parent == parent_obj:
                indicator_obj.is_deleted = False
                indicator_obj.save()
                check_child(indicator_obj)
    

    for indicator_obj in indicator_list:
        if indicator_obj.parent == indicator:
            indicator_obj.is_deleted = False
            indicator_obj.save()
            check_child(indicator_obj)


    #Parent Related Values 
    years = DataPoint.objects.all()
    for year in  years:
        try: 
           deleted_indicator = DataValue.objects.get(for_datapoint = year, for_indicator = indicator)
           deleted_indicator.is_deleted = False
           deleted_indicator.save()
        except:
            None



    messages.success(request, "Successfully Restored!")
    return HttpResponseRedirect(previous_page)


##############################
#         YEAR            #
#############################

@login_required(login_url='login')
@admin_user_required
def year_add(request, year=None):
    years = DataPoint.objects.all() # Rename sources to years
    # Fetch the largest year from the DataPoint model
    largest_year = DataPoint.objects.aggregate(Max('year_EC'))['year_EC__max'] or 0
    context = {
        'largest_year': largest_year,
        'years': years.reverse(),  # Update sources to years
    }
    return render(request, 'user-admin/add_year.html', context=context)



#############################
#        Main DashBord      #
#############################
@login_required(login_url='login')
@admin_user_required
def dashbord_topic(request):
    form = DashboardTopicForm(request.POST or None, request.FILES or None)
    topics = DashboardTopic.objects.all()
    paginator = Paginator(topics, 10) 
    page_number = request.GET.get('page')
    try:
        page = paginator.get_page(page_number)
        try: count = (10 * (int(page_number) if page_number  else 1) ) - 10
        except: count = (10 * (int(1) if page_number  else 1) ) - 10
    except PageNotAnInteger:
        # if page is not an integer, deliver the first page
        page = paginator.page(1)
        count = (10 * (int(1) if page_number  else 1) ) - 10
    except EmptyPage:
        # if the page is out of range, deliver the last page
        page = paginator.page(paginator.num_pages)
        count = (10 * (int(paginator.num_pages) if page_number  else 1) ) - 10
   
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Added')
            return redirect('dashbord_topic')
        else:
            messages.error(request, 'An error occurred while Adding')
    
    context = {
        'topics': page,
        'count' : count,
        'form' : form
    }
    return render(request, 'user-admin/dashboard_topic.html', context=context)


@admin_user_required
def dashboard_topic_delete(request, id):
    try:
        dashbord_topic = DashboardTopic.objects.get(pk = id)
        dashbord_topic.delete()
        messages.success(request, 'Successfully Deleted')
    except:
        messages.error(request, 'An error occurred while Deleting')
    
    return redirect('dashbord_topic')    


@admin_user_required
def edit_dashboard_topic(request , id):
    try:
        topic = DashboardTopic.objects.get(pk = id)
        topic.read = True
        topic.save()
    except:
        topic = None
    
    form = DashboardTopicForm(request.POST or None, request.FILES or None, instance=topic)

    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Updated')
            return redirect('dashbord_topic')
        else:
            messages.error(request, 'An error occurred while updating')
    context = {
       
        'form' : form
    }
    return render(request , "user-admin/edit_dashboard_topic.html" , context)        
    

###Topic Catagory
@login_required(login_url='login')
@admin_user_required
def topic_category(request , id):
    form = catagoryFormTopic(request.POST or None, request.FILES or None)
    topic = DashboardTopic.objects.get(id = id)
    categories = Category.objects.filter(dashboard_topic__id = id)
    paginator = Paginator(categories, 10) 
    page_number = request.GET.get('page')
    try:
        page = paginator.get_page(page_number)
        try: count = (10 * (int(page_number) if page_number  else 1) ) - 10
        except: count = (10 * (int(1) if page_number  else 1) ) - 10
    except PageNotAnInteger:
        # if page is not an integer, deliver the first page
        page = paginator.page(1)
        count = (10 * (int(1) if page_number  else 1) ) - 10
    except EmptyPage:
        # if the page is out of range, deliver the last page
        page = paginator.page(paginator.num_pages)
        count = (10 * (int(paginator.num_pages) if page_number  else 1) ) - 10
    if request.method == 'POST':
        if form.is_valid():
            form.instance.dashboard_topic = topic
            form.save()
            messages.success(request, 'Successfully Added')
            return redirect('topic_category' , id=id)
        else:
            messages.error(request, 'An error occurred while Adding')
    
    context = {
        'categories': page,
        'count' : count,
        'form' : form
    }
    return render(request, 'user-admin/topic_category.html', context=context)


@admin_user_required
def dashboard_category_delete(request, id):
    try:
        category = Category.objects.get(pk = id)
        topic = category.dashboard_topic.id
        category.delete()
        messages.success(request, 'Successfully Deleted')
    except:
        messages.error(request, 'An error occurred while Deleting')
    
    return redirect('topic_category' , topic)    


@admin_user_required
def edit_dashboard_topic_category(request , id):
    try:
        catagory = Category.objects.get(pk = id)
        catagory.read = True
        catagory.save()
    except:
        catagory = None
    
    form = catagoryFormTopic(request.POST or None, request.FILES or None, instance=catagory)
    topic = catagory.dashboard_topic.id

    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Updated')
            return redirect('topic_category' , f'{topic}')
        else:
            messages.error(request, 'An error occurred while updating')
    context = {
        "topic" : topic,
        'form' : form
    }
    return render(request , "user-admin/edit_dashboard_category.html" , context)  



###Topic Catagory Indicator
@login_required(login_url='login')
@admin_user_required
def topic_category_indicator(request , id):
    form = DashboardIndicatorForm(request.POST or None, request.FILES or None)
    category = Category.objects.get(id = id)
    indicators = Indicator.objects.filter(for_category__id = id)
    paginator = Paginator(indicators, 10) 
    page_number = request.GET.get('page')
    try:
        page = paginator.get_page(page_number)
        try: count = (10 * (int(page_number) if page_number  else 1) ) - 10
        except: count = (10 * (int(1) if page_number  else 1) ) - 10
    except PageNotAnInteger:
        # if page is not an integer, deliver the first page
        page = paginator.page(1)
        count = (10 * (int(1) if page_number  else 1) ) - 10
    except EmptyPage:
        # if the page is out of range, deliver the last page
        page = paginator.page(paginator.num_pages)
        count = (10 * (int(paginator.num_pages) if page_number  else 1) ) - 10
    if request.method == 'POST':
        if form.is_valid():
            title_ENG = form.cleaned_data['title_ENG']
            title_AMH = form.cleaned_data['title_AMH']
            type_of = form.cleaned_data['type_of']
            operation_type = form.cleaned_data['operation_type']
            is_public = form.cleaned_data['is_public']
            
            # Create a new DashboardIndicator instance and save it
            indicator = Indicator(
                title_ENG=title_ENG,
                title_AMH=title_AMH,
                type_of=type_of,
                is_public=is_public,
                for_category= category

            )
            indicator.save()
            
            # Perform any additional logic or redirection here
            messages.success(request, 'Successfully added.')
            return redirect('topic_category_indicator', id=id)
        else:
            messages.error(request, 'An error occurred while Adding')
    
    context = {
        'indicators': page,
        'count' : count,
        'form' : form
    }
    return render(request, 'user-admin/topic_category_indicator.html', context=context)



@admin_user_required
def dashboard_indicator_delete(request, id):
    try:
        indicator = Indicator.objects.get(pk = id)
        category = indicator.for_category.id
        indicator.delete()
        messages.success(request, 'Successfully Deleted')
    except:
        messages.error(request, 'An error occurred while Deleting')
    
    return redirect('topic_category_indicator' , category)        



@admin_user_required
def edit_dashboard_indicator(request , id):
    try:
        indicator = Indicator.objects.get(pk = id)
        indicator.read = True
        indicator.save()
    except:
        indicator = None
    
    form = IndicatorFormEdit(request.POST or None, request.FILES or None, instance=indicator)
    category = indicator.for_category.id

    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Updated')
            return redirect('topic_category_indicator' , f'{category}')
        else:
            messages.error(request, 'An error occurred while updating')
    context = {
        "category" : category,
        'form' : form
    }
    return render(request , "user-admin/edit_dashboard_indicator.html" , context)      



@login_required(login_url='login')
@admin_user_required
def project(request):
    form = ProjectForm(request.POST or None, request.FILES or None)
    projects = Project.objects.all()
    paginator = Paginator(projects, 10) 
    page_number = request.GET.get('page')
    try:
        page = paginator.get_page(page_number)
        try: count = (10 * (int(page_number) if page_number  else 1) ) - 10
        except: count = (10 * (int(1) if page_number  else 1) ) - 10
    except PageNotAnInteger:
        # if page is not an integer, deliver the first page
        page = paginator.page(1)
        count = (10 * (int(1) if page_number  else 1) ) - 10
    except EmptyPage:
        # if the page is out of range, deliver the last page
        page = paginator.page(paginator.num_pages)
        count = (10 * (int(paginator.num_pages) if page_number  else 1) ) - 10
   
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Added')
            return redirect('project')
        else:
            messages.error(request, 'An error occurred while Adding')
    
    context = {
        'projects': page,
        'count' : count,
        'form' : form
    }
    return render(request, 'user-admin/projects.html', context=context)    


@login_required(login_url='login')
@admin_user_required
def project_category(request , id):
    form = CategoryProjectForm(request.POST or None, request.FILES or None)
    projects = Project.objects.filter(for_catgory__id=id)
    paginator = Paginator(projects, 10) 
    page_number = request.GET.get('page')
    category = Category.objects.get(id=id)
    try:
        page = paginator.get_page(page_number)
        try: count = (10 * (int(page_number) if page_number  else 1) ) - 10
        except: count = (10 * (int(1) if page_number  else 1) ) - 10
    except PageNotAnInteger:
        # if page is not an integer, deliver the first page
        page = paginator.page(1)
        count = (10 * (int(1) if page_number  else 1) ) - 10
    except EmptyPage:
        # if the page is out of range, deliver the last page
        page = paginator.page(paginator.num_pages)
        count = (10 * (int(paginator.num_pages) if page_number  else 1) ) - 10
   
    if request.method == 'POST':
        if form.is_valid():
            obj = form.save(commit=False)
            obj.for_catgory = category
            obj.save()
            messages.success(request, 'Successfully Added')

            return redirect('project_category' , id)
        else:
            messages.error(request, 'An error occurred while Adding')
    
    context = {
        'projects': page,
        'count' : count,
        'form' : form
    }
    return render(request, 'user-admin/project_category.html', context=context)    


@admin_user_required
def project_delete(request, id):
    try:
        project = Project.objects.get(pk = id)
        project.delete()
        category = project.for_category
        messages.success(request, 'Successfully Deleted')
        return redirect('project_category' , category)
    except:
        messages.error(request, 'An error occurred while Deleting')
    
    return redirect('project')     


@admin_user_required
def edit_project(request , id):
    try:
        project = Project.objects.get(pk = id)
        project.read = True
        project.save()
    except:
        project = None
    
    form = ProjectForm(request.POST or None ,  instance=project)
  

    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Updated')
            return redirect('project')
        else:
            messages.error(request, 'An error occurred while updating')
    context = {
        "category" : category,
        'form' : form
    }
    return render(request , "user-admin/edit_project.html" , context)         

@login_required(login_url='login')
@admin_user_required
def variable_category(request , id):
    form = CategoryVariableForm(request.POST or None, request.FILES or None)
    variables = Variables.objects.filter(for_catgory__id=id)
    paginator = Paginator(variables, 10) 
    page_number = request.GET.get('page')
    category = Category.objects.get(id=id)
    try:
        page = paginator.get_page(page_number)
        try: count = (10 * (int(page_number) if page_number  else 1) ) - 10
        except: count = (10 * (int(1) if page_number  else 1) ) - 10
    except PageNotAnInteger:
        # if page is not an integer, deliver the first page
        page = paginator.page(1)
        count = (10 * (int(1) if page_number  else 1) ) - 10
    except EmptyPage:
        # if the page is out of range, deliver the last page
        page = paginator.page(paginator.num_pages)
        count = (10 * (int(paginator.num_pages) if page_number  else 1) ) - 10
   
    if request.method == 'POST':
        if form.is_valid():
            obj = form.save(commit=False)
            obj.for_catgory = category
            obj.save()
            messages.success(request, 'Successfully Added')

            return redirect('variable_category' , id)
        else:
            messages.error(request, 'An error occurred while Adding')
    
    context = {
        'variables': page,
        'count' : count,
        'form' : form
    }
    return render(request, 'user-admin/variable_category.html', context=context)    


@admin_user_required
def variable_delete(request, id):
    try:
        variable = Variables.objects.get(pk = id)
        category = variable.for_catgory.id
        variable.delete()
        messages.success(request, 'Successfully Deleted')
    except:
        messages.error(request, 'An error occurred while Deleting')
    
    return redirect('variable_category' , category)     


@admin_user_required
def edit_variable(request , id):
    try:
        variable = Variables.objects.get(pk = id)
        category = variable.for_catgory.id
        variable.read = True
        variable.save()
    except:
        variable = None
    
    form = VariableForm(request.POST or None ,  instance=variable)
  

    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Updated')
            return redirect('variable_category' ,category)
        else:
            messages.error(request, 'An error occurred while updating')
    context = {
        "category" : category,
        'form' : form
    }
    return render(request , "user-admin/edit_variable.html" , context)         


##############################
#          Reoprt            #
##############################
from django.shortcuts import render
from django.utils import timezone
from datetime import datetime
import pandas as pd
from UserManagement.models import CustomUser
from TimeSeriesBase.models import Category, Indicator
from .forms import ReportForm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
from django.http import JsonResponse
from django.utils import timezone
from datetime import datetime
from django.db.models import Count, Q
import os
import time


def generate_report(request):
    # Define default date range
    start_date = timezone.make_aware(datetime(2020, 1, 1))
    end_date = timezone.now()  # Current date and time

    # Fetch topics and their names
    topics = Topic.objects.values('id', 'title_ENG')
    topic_names_dict = {t['id']: t['title_ENG'] for t in topics}

    # Initialize the reports dictionary
    topic_reports = {}

    for topic_id, topic_name in topic_names_dict.items():
        # Filter categories based on the topic and date range
        categories = Category.objects.filter(topic=topic_id, created_at__range=[start_date, end_date])
        
        # Count the number of categories for the topic
        category_count = categories.count()

        # Get all indicators related to these categories
        indicators = Indicator.objects.filter(for_category__in=categories)

        # Count the number of indicators (including direct and child indicators)
        indicators_count = indicators.count() + Indicator.objects.filter(parent__in=indicators).count()

        # Prepare the report data for this topic
        topic_reports[topic_name] = {
            'category_count': category_count,
            'indicators_count': indicators_count
        }

    # Get user counts based on `is_staff`, `is_dashboard`, and `is_superuser` (admin) status
    user_counts = CustomUser.objects.aggregate(
        staff_count=Count('id', filter=Q(is_staff=True)),
        dashboard_count=Count('id', filter=Q(is_dashboard=True)),
        admin_count=Count('id', filter=Q(is_superuser=True))
    )
    
    # Prepare the final response data
    response_data = {
        'topics': topic_reports,
        'users': {
            'staff_count': user_counts['staff_count'],
            'dashboard_count': user_counts['dashboard_count'],
            'admin_count': user_counts['admin_count']
        },
        'end_date': end_date.strftime('%m/%d/%Y')  # Formatting date as mm/dd/yyyy
    }

    # Call AI API to generate insights
    ai_report = generate_ai_insights(response_data)

    # Include the AI-generated report in the response
    response_data['ai_report'] = ai_report

    return JsonResponse(response_data)  # Return JSON response with both topic and user counts and AI insights

def generate_ai_insights(data):
    api_key = 'AIzaSyAZGQ-XFZhLXnVBZ-72jH1hD8laIgM2j_c'  # Use environment variable for API key
    if not api_key:
        return "API key not found."

    api_url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
    
    headers = {
        'Content-Type': 'application/json'
    }
    # Adjust the prompt to instruct the AI to generate a structured report with paragraphs
    prompt = (
        "Generate a detailed and structured report based on the following data. "
        "The report should be formatted in paragraphs only, with headings if necessary. "
        "Provide sections for the introduction, data analysis, observations, and recommendations. "
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

################################
#          payment             #
################################
JSON_FILE_PATH = r'static/SampleExcel/db.json'
JSON_DATA_URL = 'http://localhost:8000/dashboard-api/json-data/'
@admin_user_required
def waiting_users(request):
    try:
        # Fetch the JSON data from the URL
        response = requests.get(JSON_DATA_URL)
        response.raise_for_status()  # Raise an error if the request was unsuccessful
        
        json_data = response.json()

        # Filter users who haven't paid or don't have a transaction ID
        waiting_users = [
            user for user in json_data
            if user.get('status') != 'paid' or not user.get('Transaction_Id')
        ]

        return render(request, 'user-admin/waiting_users.html', {'waiting_users': waiting_users})

    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    except ValueError as e:
        return JsonResponse({'error': 'Error decoding JSON'}, status=400)
@admin_user_required
def payment_his(request):
    try:
        # Fetch the JSON data from the URL
        response = requests.get(JSON_DATA_URL)
        response.raise_for_status()  # Raise an error if the request was unsuccessful
        
        json_data = response.json()

        # Filter users who have the status 'paid' and have a transaction ID
        paid_users = [
            user for user in json_data
            if user.get('status') == 'paid' and user.get('Transaction_Id')
        ]

        return render(request, 'user-admin/payment_his.html', {'paid_users': paid_users})

    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    except ValueError as e:
        return JsonResponse({'error': 'Error decoding JSON'}, status=400)
