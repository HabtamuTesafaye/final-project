
from django.contrib import admin
from import_export import resources, fields
from import_export.widgets import ForeignKeyWidget  # For foreignkey
from import_export.fields import Field
from import_export.admin import ImportExportModelAdmin #Admin 
from . import models
from import_export.formats.base_formats import XLS
import tablib


# Register your models here.
admin.site.register(models.Indicator_Point)
admin.site.register(models.DataPoint)
admin.site.register(models.Quarter)
admin.site.register(models.Month)
admin.site.register(models.Source)
admin.site.register(models.DashboardTopic)
# Import Export 

#Topic 
class TopicResource(resources.ModelResource):
    class Meta:
        model = models.Topic
        skip_unchanged = True
        report_skipped = True
        exclude = ( 'id', 'updated', 'created', 'is_deleted')
        import_id_fields = ('title_ENG', 'title_AMH',)


    
class TopicAdmin(ImportExportModelAdmin):
    resource_classes = [TopicResource]
    

admin.site.register(models.Topic, TopicAdmin)

def handle_uploaded_Topic_file(file):
    try:
        resource  = TopicResource()
        dataset = tablib.Dataset()

        imported_data = dataset.load(file.read())
        result = resource.import_data(imported_data, dry_run=True, collect_failed_rows = True)
        
        if not result.has_errors():
            return True, imported_data, result
        else:
            return False, imported_data, result
    except Exception as e:
         return False, imported_data, result




#Category 
class CategoryResource(resources.ModelResource):
    topic = fields.Field(
        column_name='topic',
        attribute='topic',
        widget=ForeignKeyWidget(models.Topic, field='title_ENG'),
        saves_null_values = True,
        )


    def before_import_row(self, row, **kwargs):
            if row.get('topic') is None:
               pass
            else :
               topic_name = row["topic"]
               models.Topic.objects.get_or_create(title_ENG=topic_name, defaults={"title_ENG": topic_name, "title_AMH":topic_name})
    
    class Meta:
        model = models.Category
        report_skipped = True
        skip_unchanged = True
        exclude = ( 'id', 'created_at', 'is_deleted')
        import_id_fields = ('name_ENG', 'name_AMH','topic')

class CategoryAdmin(ImportExportModelAdmin):
    resource_classes = [CategoryResource]



admin.site.register(models.Category, CategoryAdmin)

import logging
import tablib
from import_export.formats.base_formats import XLSX

# Configure logging
logger = logging.getLogger(__name__)

def handle_uploaded_Category_file(file):
    try:
        # Initialize the resource and dataset
        resource = CategoryResource()

        # Load the file as an XLSX dataset
        dataset = tablib.Dataset().load(file.read(), format='xlsx')

        # Filter out unwanted columns, only keeping the relevant ones
        relevant_columns = ['name_ENG', 'name_AMH', 'topic']
        new_data_set = []

        for row in dataset.dict:
            filtered_row = {key: row[key] for key in relevant_columns if key in row}
            new_data_set.append((filtered_row.get('name_ENG', ''),
                                 filtered_row.get('name_AMH', ''),
                                 filtered_row.get('topic', '')))

        # Create a new dataset with only the relevant data
        dataset = tablib.Dataset(*new_data_set, headers=relevant_columns)

        # Perform a dry run to check for errors
        result = resource.import_data(dataset, dry_run=True)

        if not result.has_errors():
            logger.info("File imported successfully with no errors.")
            return True, dataset, result
        else:
            logger.warning("Errors occurred during the import.")
            return False, dataset, result

    except Exception as e:
        logger.error(f"Error during import: {str(e)}", exc_info=True)
        return False, '', ''
       

#Indicator   
import datetime

# Get the current date and time
current_datetime = datetime.datetime.now()

class IndicatorResource(resources.ModelResource):
    for_category = fields.Field(
        column_name='for_category',
        attribute='for_category',
        widget=ForeignKeyWidget(models.Category, field='name_ENG'),
        saves_null_values=True,
    )
    
    parent = fields.Field(
        column_name='parent',
        attribute='parent',
        widget=ForeignKeyWidget(models.Indicator, field='id'),
        saves_null_values=True,
    )

    measurement = fields.Field(
        column_name='measurement',
        attribute='measurement',
        widget=ForeignKeyWidget(models.Measurement, field='Amount_ENG'),
        saves_null_values=True,
    )

    def before_save_instance(self, instance, *args, **kwargs):
        # Set the created_at field to the current datetime
        instance.created_at = datetime.datetime.now()

    class Meta:
        model = models.Indicator
        report_skipped = True
        skip_unchanged = True
        fields = ('id', 'parent', 'title_ENG', 'title_AMH', 'for_category', 'measurement', 'type_of')
        exclude = ('created_at', 'is_deleted', 'composite_key', 'op_type')
        import_id_fields = ('parent', 'title_ENG', 'title_AMH', 'for_category', 'measurement', 'type_of')
        export_order = ('id', 'parent', 'title_ENG', 'title_AMH', 'for_category', 'measurement', 'type_of')

class Indicatoradmin(ImportExportModelAdmin):
    resource_classes = [IndicatorResource]
    search_fields = ['title_ENG', 'title_AMH']



def handle_uploaded_Indicator_file(file, category):
        def filterParent(item):
            '''
            filter parent Items
            '''
            if item['parent'] == None:
                return item
            

        def filterChild(itemParent, itemChild):
            '''
            filter Child Items
            '''
            try:
               if itemChild['parent'].strip() == itemParent.strip():
                   return itemChild
            except:
               pass
           
           
           

  
        resource  = IndicatorResource()
        dataset = tablib.Dataset()


        imported_data = dataset.load(file.read())
        total_data = []
            
        indicator_list = []
        for row in imported_data.dict:
            if row['parent'] == None and row['title_ENG'] != None :
                indicator_list.append({'parent':row['parent'], 'title_ENG':row['title_ENG'],  'title_AMH':row['title_AMH'],'for_category':category.name_ENG,'measurement': row['measurement'],'type_of':row['type_of']})
            elif row['parent'] != None:
                indicator_list.append({'parent':row['parent'], 'title_ENG':row['title_ENG'],  'title_AMH':row['title_AMH'],'for_category':None,'measurement': None,'type_of':None})
        
        parentIndicator = list(filter(lambda parent_item: filterParent(parent_item), indicator_list ))

        #Child 
        global current_id 
        current_id = 0
        def filterChildIndicator(parent_id, parent_name):
            global current_id
            childIndicator = filter(lambda child_item : filterChild(parent_name, child_item), indicator_list )
            childIndicator =  list(childIndicator)

            if len(childIndicator) != 0:
                for child in childIndicator:                    
                    data = (parent_id,f'{child["title_ENG"].strip()}' ,child["title_AMH"],None, None,None)
                    child_dataset = tablib.Dataset(data, headers=['parent', 'title_ENG', 'title_AMH', 'for_category', 'measurement', 'type_of'])
                    result = resource.import_data(child_dataset, dry_run=True)
                    if not result.has_errors():
                        current_id = int(current_id) + 1
                        total_data.append(((int(current_id), parent_id,f"{child['title_ENG'].strip()}" ,child['title_AMH'],None, None,None)))
                        filterChildIndicator(int(current_id), child['title_ENG'])
                    else:
                        current_id = int(current_id) + 1
                        total_data.append((current_id, parent_id,f"{child['title_ENG'].strip()}" ,child['title_AMH'],None, None,None))
                        filterChildIndicator(int(current_id), child['title_ENG'])
                             
                
        #Parent  
        for parent in parentIndicator:
            data = (parent['parent'],f"{parent['title_ENG'].strip()}",parent['title_AMH'],category.name_ENG, parent['measurement'],parent['type_of'].strip())
            parent_dataset = tablib.Dataset(data, headers=['parent', 'title_ENG', 'title_AMH', 'for_category', 'measurement', 'type_of'])
            result = resource.import_data(parent_dataset, dry_run=True)
            if not result.has_errors():
                for row_result in result:
                    get_id = ("%d" % (row_result.object_id))
                parent_id = get_id
                if current_id == 0:
                    current_id = int(current_id) + int(parent_id)
                else:
                    current_id = int(current_id) + 1
                total_data.append((current_id, None,f"{parent['title_ENG'].strip()}",parent['title_AMH'],category.name_ENG, parent['measurement'],parent['type_of'].strip()))
                filterChildIndicator(current_id, parent['title_ENG'])
            else:
                current_id = int(current_id) + 1
                total_data.append((current_id, None,f"{parent['title_ENG'].strip()}",parent['title_AMH'],category.name_ENG, parent['measurement'],parent['type_of'].strip()))
                filterChildIndicator(int(current_id), parent['title_ENG'])

        

        #Return the data
        data_set = tablib.Dataset(*total_data, headers=['id','parent', 'title_ENG', 'title_AMH', 'for_category', 'measurement', 'type_of'])
        print(data_set)
        result = resource.import_data(data_set, dry_run=True)
        return True,data_set, result



    

admin.site.register(models.Indicator, Indicatoradmin)



#Measurement  
class MeasurementResource(resources.ModelResource):    
    parent = fields.Field(
        column_name='parent',
        attribute='parent',
        widget=ForeignKeyWidget(models.Measurement, field='Amount_ENG'),
        saves_null_values = True,
    )

    class Meta:
        model = models.Measurement
        skip_unchanged = True
        report_skipped = True
        exclude = ( 'id', 'updated', 'created', 'is_deleted')
        import_id_fields = ('Amount_ENG', 'Amount_AMH','parent')
        export_order = ('parent','Amount_ENG', 'Amount_AMH')
        

class MeasurementAdmin(ImportExportModelAdmin):
    resource_classes = [MeasurementResource]


def handle_uploaded_Measurement_file(file):
    try:
        resource  = MeasurementResource()
        dataset = tablib.Dataset()

        imported_data = dataset.load(file.read())
        result = resource.import_data(imported_data, dry_run=True)
        if not result.has_errors():
            return True, imported_data, result
        else:
            return False,imported_data, result
    except Exception as e:
        return False, '', ''
    

admin.site.register(models.Measurement, MeasurementAdmin)



#Value
class DataValueResource(resources.ModelResource):    
    for_indicator = fields.Field(
        column_name='for_indicator',
        attribute='for_indicator',
        widget=ForeignKeyWidget(models.Indicator, field='composite_key'),
        saves_null_values = True,
    ) 
    

    for_datapoint = fields.Field(
        column_name='for_datapoint',
        attribute='for_datapoint',
        widget=ForeignKeyWidget(models.DataPoint, field='year_EC'),
        saves_null_values = True,
    )

    for_quarter = fields.Field(
        column_name='for_quarter',
        attribute='for_quarter',
        widget=ForeignKeyWidget(models.Quarter, field='title_ENG'),
        saves_null_values = True,
    )

    for_month = fields.Field(
        column_name='for_month',
        attribute='for_month',
        widget=ForeignKeyWidget(models.Month, field='title_ENG'),
        saves_null_values = True,
    )

    class Meta:
        model = models.DataValue
        skip_unchanged = True
        report_skipped = True
        use_bulk = True
        exclude = ( 'id', 'is_deleted','for_source')
        fields = ('for_indicator', 'for_datapoint', 'for_quarter', 'for_month', 'value', )
        import_id_fields = ('for_datapoint', 'for_quarter', 'for_month', 'value', 'for_indicator')
        export_order = ('for_indicator', 'for_datapoint', 'for_quarter', 'for_month', 'value', )
    


        

class DataValueAdmin(ImportExportModelAdmin):
    resource_classes = [DataValueResource]


def handle_uploaded_DataValue_file(file, type_of_data):
    if type_of_data == 'yearly':
        try:
            resource  = DataValueResource()
            dataset = tablib.Dataset()
            imported_data = dataset.load(file.read())
    
            data_set = []
            for item in imported_data.dict:
                for i, key in enumerate(list(item.keys())):
                    if i != 0:
                         data_set.append((item['for_indicator'].lstrip(), key, None, None,  round(item[key], 1) if item[key] else 0 ))
    
            data_set_table = tablib.Dataset(*data_set, headers=['for_indicator', 'for_datapoint', 'for_quarter', 'for_month','value'])
            
            result = resource.import_data(data_set_table, dry_run=True)
            return True, data_set_table, result
        except Exception as e:
            return False, '', ''
   
    elif type_of_data == 'monthly':
        try:
            resource  = DataValueResource()
            dataset = tablib.Dataset()
            imported_data = dataset.load(file.read())
            data_set = []
            for item in imported_data.dict:
                for i, key in enumerate(list(item.keys())):
                    if i != 0 and i != 1:
                        data_set.append((item['Year'], item['Month'] ,key.strip(), round(item[key] ,1) if item[key]  else 0  ))

            data_set_table = tablib.Dataset(*data_set, headers=['for_datapoint','for_month','for_indicator', 'value'])
            result = resource.import_data(data_set_table, dry_run=True)
      
        
            return True, data_set_table, result
        except Exception as e:
            return False, '', ''

    elif type_of_data == 'quarterly':
        try:    
            resource  = DataValueResource()
            dataset = tablib.Dataset()
            imported_data = dataset.load(file.read())
    
            data_set = []
            for item in imported_data.dict:
                for i, key in enumerate(list(item.keys())):
                    if i != 0 and i != 1:
                        data_set.append((item['Year'], item['Quarter'] ,key.strip(), round(item[key] ,1) if item[key] else 0  ))
    
            data_set_table = tablib.Dataset(*data_set, headers=['for_datapoint','for_quarter','for_indicator', 'value'])
            result = resource.import_data(data_set_table, dry_run=True)
            return True, data_set_table, result
        except Exception as e:
            return False, '', ''
    

admin.site.register(models.DataValue, DataValueAdmin)




###Confirm 
def confirm_file(imported_data, type):
    try:
        if type == 'topic':
            resource  = TopicResource()
        elif type == 'category':
            resource = CategoryResource()
        elif type == 'measuremennt':
            resource = MeasurementResource()
        elif type == 'indicator':
            resource = IndicatorResource()
        elif type == 'data_value':
            resource = DataValueResource()
        result = resource.import_data(imported_data, dry_run=True, collect_failed_rows = True)
        
        if not result.has_errors():
            resource.import_data(imported_data, dry_run=False)  # Actually import now
            return True, f"Data imported successfully: {len(imported_data)} records imported."
        else:
            return False, f"Error importing data: Please review your Dcoument."
    except Exception as e:
         return False, f"Error importing data: Please review your Document."
        

##CKeditor

from ckeditor.widgets import CKEditorWidget


from django import forms

class ProjectAdminForm(forms.ModelForm):
    content = forms.CharField(widget=CKEditorWidget())
    class Meta:
        model = models.Project
        fields = '__all__'

class ProjectAdmin(admin.ModelAdmin):
    form = ProjectAdminForm

admin.site.register(models.Project, ProjectAdmin)