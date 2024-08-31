from datetime import datetime, timedelta
from django.db import models
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.db import models
from auditlog.registry import auditlog
from decimal import Decimal
from fontawesome_5.fields import IconField
from django.contrib.postgres.fields import JSONField
from ckeditor.fields import RichTextField


class SiteConfiguration(models.Model):
    is_public = models.BooleanField(default=False)

    def __str__(self):
        return "Public" if self.is_public else "Private"



class Project(models.Model):
    title_ENG = models.CharField(max_length=300, unique = True)
    title_AMH = models.CharField(max_length=300, null = True)
    for_catgory = models.ForeignKey("Category" , on_delete=models.SET_NULL, null=True,blank=True)
    content = RichTextField()
    is_dashboard_visible = models.BooleanField(default = False)

    def __str__(self):
        return self.title_ENG 



class Variables(models.Model):
    title_ENG = models.CharField(max_length=300, unique = True)
    title_AMH = models.CharField(max_length=300, null = True)
    for_catgory = models.ForeignKey("Category" , on_delete=models.SET_NULL, null=True,blank=True)
    content = RichTextField()
    is_dashboard_visible = models.BooleanField(default = False)

    def __str__(self):
        return self.title_ENG 


class Topic(models.Model):
    title_ENG = models.CharField(max_length=300, unique = True)
    title_AMH = models.CharField(max_length=300, null = True)
    updated =  models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)

    def __str__(self):
        return self.title_ENG


class DashboardTopic(models.Model):
    title_ENG = models.CharField(max_length=300, unique = True)
    title_AMH = models.CharField(max_length=300, null = True)
    icon = IconField()
    updated =  models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)
    order = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.title_ENG + " : Order : " + str(self.order)
    
    class Meta:
        ordering = ['order'] #Oldest First
    
class Category(models.Model):
    name_ENG = models.CharField(max_length=300, unique = True)
    name_AMH = models.CharField(max_length=300, unique = True)
    dashboard_topic = models.ForeignKey(DashboardTopic, null=True, blank=True, on_delete=models.SET_NULL)
    topic = models.ForeignKey(Topic, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    is_deleted = models.BooleanField(default=False)

    def __str__(self):
        return self.name_ENG

data_point_type = [
    ('yearly', 'Yearly'),
    ('quarterly', 'Quarterly'),
    ('monthly', 'Monthly'),
]

operation_type = [
    ('none', 'none'),
    ('sum', 'Sum'),
    ('average','Average')
]



class Indicator(models.Model):
    title_ENG = models.CharField(max_length=300) 
    title_AMH = models.CharField(max_length=300 , null=True, blank=True)
    composite_key = models.CharField(max_length=300, unique = True)
    op_type = models.CharField(max_length = 60, choices = operation_type,default = 'sum') 
    parent = models.ForeignKey('self', related_name='children', on_delete=models.CASCADE, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    for_category = models.ForeignKey(Category, blank=True, null=True, on_delete=models.SET_NULL)
    is_deleted = models.BooleanField(default = False)
    measurement = models.ForeignKey('Measurement', blank=True, null=True, on_delete=models.CASCADE)
    type_of = models.CharField(choices=data_point_type ,max_length=60, null=True, blank=True)
    is_public = models.BooleanField(default = True)
    is_dashboard_visible = models.BooleanField(default = False)
   
    def save(self, *args, **kwargs):
        self.composite_key = str(self.title_ENG.replace(" ","").replace("/","").replace("&","")) +  str(self.id)
        super(Indicator, self).save(*args, **kwargs)

    def __str__(self):
        return self.title_ENG 



class Indicator_Point(models.Model):
    is_actual = models.BooleanField()
    for_datapoint = models.ForeignKey("DataPoint",on_delete=models.SET_NULL, null = True)
    for_indicator = models.ForeignKey(Indicator,on_delete=models.SET_NULL, null = True)
   
    
    def __str__(self):
         return str(self.for_indicator.title_ENG ) +  " Actual: " + str(self.is_actual) + " Year: " + str(self.for_datapoint)

class DataPoint(models.Model):
    year_EC = models.CharField(max_length=50, null=True, blank=True, unique=True)
    year_GC = models.CharField(max_length=50, null=True, blank=True, unique = True)
    is_interval = models.BooleanField(default=False)
    year_start_EC = models.CharField(max_length=50, null=True, blank=True)
    year_start_GC = models.CharField(max_length=50, null=True, blank=True)
    year_end_EC = models.CharField(max_length=50, null=True, blank=True)
    year_end_GC = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now_add=True)
    
    
    class Meta:
        ordering = ['year_EC'] #Oldest First
        

    def save(self, *args, **kwargs):
        self.year_GC = f'{str(int(self.year_EC )+ 7)}/{str(int(self.year_EC)+ 8)}'
        super(DataPoint, self).save(*args, **kwargs)


    def __str__(self):
        if self.is_interval:
            return self.year_start_EC + " - " + self.year_end_EC + "E.C"
        else:
            return self.year_EC+" "+"E.C"
    
class Quarter(models.Model):
    title_ENG = models.CharField(max_length=50)
    title_AMH = models.CharField(max_length=50)
    number = models.IntegerField()
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at'] #Oldest First
    
    def __str__(self):
        return self.title_AMH + " " + self.title_AMH
    
class Month(models.Model):
    month_ENG = models.CharField(max_length=50)
    month_AMH = models.CharField(max_length=50)
    number = models.IntegerField()
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['number'] #Oldest First

    def __str__(self):
        return self.month_AMH + " : " + self.month_ENG + " ==> " + str(self.number)
    
class Measurement(models.Model):
    Amount_ENG = models.CharField(max_length=50)
    Amount_AMH = models.CharField(max_length=50, null=True, blank=True)
    parent = models.ForeignKey('self', related_name='children', on_delete=models.CASCADE, blank=True, null=True)
    is_deleted = models.BooleanField(default = False)
    updated =  models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    

    def get_full_path(self):
        full_path = [self.Amount_ENG]
        k = self.parent
        while k is not None:
            full_path.append(k.Amount_ENG)
            k = k.parent
        return ' -> '.join(full_path[::-1])

    def __str__(self):
        return self.get_full_path()
    

value_entrance_type = [
    ('1month', '1 Month'),
    ('2month','2 Month'),
    ('3month','3 Month'),
    ('4month','4 Month'),
    ('5month','5 Month'),
    ('6month','6 Month'),
    ('7month','7 Month'),
    ('8month','8 Month'),
    ('9month','9 Month'),
    ('10month','10 Month'),
    ('11month','11 Month'),
]


class DataValue(models.Model):
    value = models.FloatField(blank=True ,null=True, max_length=40)
    value_entrance_type = models.CharField(max_length=15, choices=value_entrance_type, null=True, blank=True)
    for_quarter = models.ForeignKey("Quarter", on_delete=models.SET_NULL, blank=True ,null=True)
    for_month = models.ForeignKey("Month", on_delete=models.SET_NULL, blank=True ,null=True)
    for_datapoint = models.ForeignKey("DataPoint", related_name="datavalue_set", on_delete=models.SET_NULL, blank=True, null=True)
    for_source = models.ForeignKey("Source",on_delete=models.SET_NULL  ,blank=True, null=True,)
    for_indicator = models.ForeignKey(Indicator,related_name='indicator_value' ,null=True, blank=True, on_delete=models.SET_NULL)
    is_deleted = models.BooleanField(default = False)


    def __str__(self) -> str:
        return str(self.for_indicator)
    
    def save(self, *args, **kwargs):
        self.value = round(self.value,1)
        super(DataValue, self).save(*args, **kwargs)
    
    
    def calculate_parent_value(self):
        try: 
            #Year Type Data
            if  (not self.for_month and self.for_datapoint) and (not self.for_quarter and self.for_datapoint) :
                main_parent = self.for_indicator.parent # get Parent of Indicator
                try: parent_data_value = DataValue.objects.get(for_indicator = main_parent, for_datapoint = self.for_datapoint)
                except:parent_data_value = None
                child_indicators = Indicator.objects.filter(parent = main_parent, is_deleted = False)
              
               
               
                if main_parent.op_type == 'sum' or main_parent.op_type == None:
                    sum = 0
                    for child in child_indicators:
                        try: 
                            child_data_value = DataValue.objects.get(for_indicator = child, for_datapoint = self.for_datapoint)
                            if(child_data_value.is_deleted == False):
                                sum  = sum + Decimal(child_data_value.value)
                            else:
                                None
                        except: 
                            None
                        
                    if parent_data_value is None and main_parent is not None:
                        parent_data = DataValue()
                        parent_data.value = round(sum,1)
                        parent_data.for_indicator = main_parent
                        parent_data.for_datapoint = self.for_datapoint
                        parent_data.save()
                    else: 
                        parent_data_value.value = round(sum, 1)
                        try: parent_data_value.save()
                        except: None
                elif main_parent.op_type == 'average':
                    avg = 0
                    for child in child_indicators:
                        try: 
                            child_data_value = DataValue.objects.get(for_indicator = child, for_datapoint = self.for_datapoint)
                            if(child_data_value.is_deleted == False):
                                avg  = avg + Decimal(child_data_value.value)
                            else:
                                None
                        except: 
                            None
                        
                    if parent_data_value is None and main_parent is not None:
                        parent_data = DataValue()
                        parent_data.value = round(avg/len(child_indicators),1)
                        parent_data.for_indicator = main_parent
                        parent_data.for_datapoint = self.for_datapoint
                        parent_data.save()
                    else: 
                        parent_data_value.value = round(avg/len(child_indicators), 1)
                        try: parent_data_value.save()
                        except: None



            #Month Type Data 
            elif not self.for_quarter and self.for_datapoint :
                main_parent = self.for_indicator.parent
                try: parent_data_value = DataValue.objects.get(for_indicator = main_parent, for_datapoint = self.for_datapoint, for_month = self.for_month)
                except:parent_data_value = None
                child_indicators = Indicator.objects.filter(parent = main_parent, is_deleted = False)

                if main_parent.op_type == 'sum' or main_parent.op_type == None:
                    sum = 0
                    for child in child_indicators:
                       try: 
                           child_data_value = DataValue.objects.get(for_indicator = child, for_datapoint = self.for_datapoint , for_month = self.for_month)
                           if(child_data_value.is_deleted == False):
                               sum  = sum + Decimal(child_data_value.value)
                           else:
                               None
                       except: 
                           None
                    if parent_data_value is None and main_parent is not None:
                       parent_data = DataValue()
                       parent_data.value = round(sum,1)
                       parent_data.for_indicator = main_parent
                       parent_data.for_datapoint = self.for_datapoint
                       parent_data.for_month = self.for_month
                       parent_data.save()
                    elif parent_data_value is not None: 
                       parent_data_value.value = round(sum,1)
                       try: parent_data_value.save()
                       except: None
                elif main_parent.op_type == 'average':
                    avg = 0
                    for child in child_indicators:
                       try: 
                           child_data_value = DataValue.objects.get(for_indicator = child, for_datapoint = self.for_datapoint , for_month = self.for_month)
                           if(child_data_value.is_deleted == False):
                               avg  = avg + Decimal(child_data_value.value)
                           else:
                               None
                       except: 
                           None
                    if parent_data_value is None and main_parent is not None:
                       parent_data = DataValue()
                       parent_data.value = round(avg/len(child_indicators),1)
                       parent_data.for_indicator = main_parent
                       parent_data.for_datapoint = self.for_datapoint
                       parent_data.for_month = self.for_month
                       parent_data.save()
                    elif parent_data_value is not None: 
                       parent_data_value.value = round(avg/len(child_indicators),1)
                       try: parent_data_value.save()
                       except: None


            #Quarter Type Data
            elif  not self.for_month and self.for_datapoint :
                main_parent = self.for_indicator.parent
                try: parent_data_value = DataValue.objects.get(for_indicator = main_parent, for_datapoint = self.for_datapoint, for_quarter = self.for_quarter)
                except:parent_data_value = None
                child_indicators = Indicator.objects.filter(parent = main_parent, is_deleted = False)

                if main_parent.op_type == 'sum' or main_parent.op_type == None:
                    sum = 0
                    for child in child_indicators:
                       try: 
                           child_data_value = DataValue.objects.get(for_indicator = child, for_datapoint = self.for_datapoint , for_quarter = self.for_quarter)
                           if(child_data_value.is_deleted == False):
                               sum  = sum + Decimal(child_data_value.value)
                           else:
                               None
                       except: 
                           None
                    if parent_data_value is None and main_parent is not None:
                       parent_data = DataValue()
                       parent_data.value = round(sum,1)
                       parent_data.for_indicator = main_parent
                       parent_data.for_datapoint = self.for_datapoint
                       parent_data.for_quarter = self.for_quarter
                       parent_data.save()
                    else: 
                       parent_data_value.value = round(sum,1)
                       try: parent_data_value.save()
                       except: None
                elif main_parent.op_type == 'average':
                    avg = 0
                    for child in child_indicators:
                       try: 
                           child_data_value = DataValue.objects.get(for_indicator = child, for_datapoint = self.for_datapoint , for_quarter = self.for_quarter)
                           if(child_data_value.is_deleted == False):
                               avg  = avg + Decimal(child_data_value.value)
                           else:
                               None
                       except: 
                           None
                    if parent_data_value is None and main_parent is not None:
                       parent_data = DataValue()
                       parent_data.value = round(avg/len(child_indicators),1)
                       parent_data.for_indicator = main_parent
                       parent_data.for_datapoint = self.for_datapoint
                       parent_data.for_quarter = self.for_quarter
                       parent_data.save()
                    else: 
                       parent_data_value.value = round(avg/len(child_indicators),1)
                       try: parent_data_value.save()
                       except: None        
        except:
            None

@receiver(post_save, sender=DataValue)
def call_my_function(sender, instance, created, **kwargs):
    if created: 
        instance.calculate_parent_value()
    else:  
        instance.calculate_parent_value()

class Source(models.Model):
    title_ENG = models.CharField(max_length=50)
    title_AMH = models.CharField(max_length=50)
    updated =  models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default = False)
    def __str__(self):
        return self.title_ENG


auditlog.register(Topic)
auditlog.register(Category)
auditlog.register(Indicator)
auditlog.register(Indicator_Point)
auditlog.register(DataPoint)
auditlog.register(Quarter)
auditlog.register(Month)
auditlog.register(Measurement)
auditlog.register(DataValue)
auditlog.register(Source)

