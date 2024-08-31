from django import forms
from TimeSeriesBase.models import Topic,Category,Source,Measurement, Indicator, DataPoint, Month, DataValue,SiteConfiguration , DashboardTopic ,Project ,Variables

class ReportForm(forms.Form):
    topic = forms.ModelChoiceField(
        queryset=Topic.objects.all(),
        required=True,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    start_date = forms.DateField(
        required=True,
        widget=forms.TextInput(attrs={'type': 'date', 'class': 'form-control'})
    )
    end_date = forms.DateField(
        required=True,
        widget=forms.TextInput(attrs={'type': 'date', 'class': 'form-control'})
    )

class SiteConfigurationForm(forms.ModelForm):
    class Meta:
        model = SiteConfiguration
        fields = ['is_public']
        widgets = {
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
        initial = {
            'is_public': False,  # Set the initial value to False
        }

data_point_type = [
    ('yearly', 'Yearly'),
    ('quarterly', 'Quarterly'),
    ('monthly', 'Monthly'),
]


class ImportFileIndicatorForm(forms.Form):
    category = forms.ModelChoiceField(queryset=Category.objects.all(), widget=forms.Select(attrs={
        'class' : 'form-select'
    }))
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={
        'class' : 'form-control'
    }))

class ImportFileForm(forms.Form):
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={
        'class' : 'form-control'
    }))



class ImportFileIndicatorAddValueForm(forms.Form):
    type_of_data = forms.ChoiceField(required=True, choices=data_point_type, widget=forms.Select(attrs={
        'class' : 'form-select'
    }))
    file = forms.FileField(required=True,widget=forms.ClearableFileInput(attrs={
        'class' : 'form-control'
    }))

class ImportFileForm(forms.Form):
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={
        'class' : 'form-control'
    }))

class catagoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ('name_ENG', 'name_AMH', 'topic')

        widgets = {
            'name_ENG': forms.TextInput(attrs={'class': 'form-control'}),
            'name_AMH': forms.TextInput(attrs={'class': 'form-control'}),
            'topic': forms.Select(attrs={'class': 'form-select'}),
        }

    def __init__(self, *args, **kwargs):
        super(catagoryForm, self).__init__(*args, **kwargs)
        # Override the queryset for the topic field
        self.fields['topic'].queryset = Topic.objects.filter(is_deleted=False)

operation_type = [
    ('sum', 'Sum'),
    ('average','Average')
]




class catagoryFormTopic(forms.ModelForm):
    class Meta:
        model = Category
        fields = ('name_ENG', 'name_AMH')

        widgets = {
            'name_ENG': forms.TextInput(attrs={'class': 'form-control'}),
            'name_AMH': forms.TextInput(attrs={'class': 'form-control'}),
            
        }

  



class IndicatorForm(forms.Form):
    data_point_type = [
    ('yearly', 'Yearly'),
    ('quarterly', 'Quarterly'),
    ('monthly', 'Monthly'),
]
    title_ENG = forms.CharField(required=True, widget=forms.TextInput(attrs={
        'class' : 'form-control'
    }))
    title_AMH = forms.CharField(required=False, widget=forms.TextInput(attrs={
        'class' : 'form-control'
    })) 
    for_category = forms.ModelChoiceField(queryset=Category.objects.all(), required=True, widget=forms.Select(attrs={'class': 'form-select','data-placeholder' : "Select Category"}))
    type_of = forms.CharField(required=True, widget=forms.Select(choices=data_point_type,attrs={
        'class' : 'form-select'
    }))
    operation_type = forms.ChoiceField(required=True, choices = operation_type ,widget=forms.Select(attrs={
        'class' : 'form-select'
    }))
    is_public = forms.BooleanField(required=False)
    



class DashboardIndicatorForm(forms.Form):
    data_point_type = [
    ('yearly', 'Yearly'),
    ('quarterly', 'Quarterly'),
    ('monthly', 'Monthly'),
]
    title_ENG = forms.CharField(required=True, widget=forms.TextInput(attrs={
        'class' : 'form-control'
    }))
    title_AMH = forms.CharField(required=False, widget=forms.TextInput(attrs={
        'class' : 'form-control'
    })) 
    type_of = forms.CharField(required=True, widget=forms.Select(choices=data_point_type,attrs={
        'class' : 'form-select'
    }))
    operation_type = forms.ChoiceField(required=True, choices = operation_type ,widget=forms.Select(attrs={
        'class' : 'form-select'
    }))
    is_public = forms.BooleanField(required=False)    
    is_dashboard_visible = forms.BooleanField(required=False)

class IndicatorFormEdit(forms.ModelForm):
    class Meta:
        model = Indicator
        fields = ['title_ENG', 'title_AMH', 'parent', 'is_deleted', 'measurement', 'type_of', 'is_public', 'is_dashboard_visible']
        widgets = {
            'title_ENG': forms.TextInput(attrs={'class': 'form-control'}),
            'title_AMH': forms.TextInput(attrs={'class': 'form-control'}),
            'parent': forms.Select(attrs={'class': 'form-control'}),
            'is_deleted': forms.CheckboxInput(attrs={'class': 'form-check-input ml-3'}),
            'measurement': forms.Select(attrs={'class': 'form-control'}),
            'type_of': forms.Select(attrs={'class': 'form-control'}),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input ml-3'}),
            'is_dashboard_visible': forms.CheckboxInput(attrs={'class': 'form-check-input ml-3'}),
        }


class operationForm(forms.Form):
     operation_type = forms.ChoiceField(required=True, choices = operation_type ,widget=forms.Select(attrs={
        'class' : 'form-select'
    }))






class IndicatorSubForm(forms.Form):
    data_point_type = [
    ('yearly', 'Yearly'),
    ('quarterly', 'Quarterly'),
    ('monthly', 'Monthly'),
]
    title_ENG = forms.CharField(required=True, widget=forms.TextInput(attrs={
        'class' : 'form-control'
    }))
    title_AMH = forms.CharField(required=False, widget=forms.TextInput(attrs={
        'class' : 'form-control'
    })) 
    operation_type = forms.ChoiceField(required=True, choices = operation_type ,widget=forms.Select(attrs={
        'class' : 'form-select'
    }))
    is_public = forms.BooleanField(required=False)



class SubIndicatorForm(forms.Form):
    title_ENG_add = forms.CharField(widget=forms.TextInput(attrs={
        'class' : 'form-control'
    }))
    title_AMH_add = forms.CharField(required=False, widget=forms.TextInput(attrs={
        'class' : 'form-control'
    }))
    is_public = forms.BooleanField(required=False)

class SubIndicatorFormDetail(forms.ModelForm):
    class Meta:
        model = Indicator
        fields =  ('title_ENG', 'title_AMH')
        
        widgets = {
            'title_ENG' : forms.TextInput(attrs={
                'class' : 'form-control'
            }),
            'title_AMH' : forms.TextInput(attrs={
                'class' : 'form-control'
            }),
        }

class TopicForm(forms.ModelForm):
    class Meta:
        model = Topic
        fields = ('title_ENG', 'title_AMH')

        widgets = {
            'title_ENG': forms.TextInput(attrs={
                'class': 'form-control'
            }),
            'title_AMH': forms.TextInput(attrs={
                'class': 'form-control'
            })
        }

class SourceForm(forms.ModelForm):
    class Meta:
        model = Source
        fields = ('title_ENG', 'title_AMH')

        widgets = {
                'title_ENG': forms.TextInput(attrs={
                    'class': 'form-control'
                }),
                'title_AMH': forms.TextInput(attrs={
                    'class': 'form-control'
                })
        }

class YearForm(forms.ModelForm):
    class Meta:
        model = DataPoint
        fields = ('year_EC',)

        widgets = {
            'year_EC': forms.TextInput(attrs={
                'class': 'form-control'
            })
        }

class MeasurementForm(forms.ModelForm):
    class Meta:

        model = Measurement
        fields = ('Amount_ENG', 'Amount_AMH' )
    
        widgets = {
                'Amount_ENG': forms.TextInput(attrs={
                    'class': 'form-control'
                }),
                'Amount_AMH': forms.TextInput(attrs={
                    'class': 'form-control'
                })
        }

class MeasurementSelectForm(forms.Form):
    measurement_form = forms.CharField(widget=forms.Select(attrs={
        'class' : 'form-control'
    }))

class DataPointForm(forms.ModelForm):
    class Meta:
        model = DataPoint
        fields = '__all__'
        
        widgets = {
            'year_EC' : forms.DateInput(attrs={
                'class' : 'form-control',
                'type' : 'number',
                'placeholder':'Please Enter Year E.C (Required for Non-Interval) ',
                'min' : '1900',
            }),
            'year_start_EC' : forms.DateInput(attrs={
                'class' : 'form-control',
                'type' : 'number',
                'placeholder':'Please Enter Year (Not Required, For Interval Year)',
                'min' : '1900',
            }),
            'year_end_EC' : forms.DateInput(attrs={
                'class' : 'form-control',
                'type' : 'number',
                'placeholder':'Please Enter Year (Not Required, For Interval Year)',
                'min' : '1900',
            }),
            'is_interval' : forms.CheckboxInput(attrs={
                'class' : 'form-check'
            })
        }
        
class dataListForm(forms.Form):
    topic = forms.ModelChoiceField(queryset=Topic.objects.all(),required=True,widget=forms.Select(attrs={
        'class' : 'form-control'
    }))
    category = forms.ModelChoiceField(queryset=Category.objects.all(),required=True,widget=forms.Select(attrs={
        'class' : 'form-control'
    }))
    is_interval = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={
        'class' : 'form-check'
    }))
    year  = forms.ModelChoiceField(queryset=DataPoint.objects.all(),required=True,widget=forms.Select(attrs={
        'class' : 'form-control'
    }))
    indicator = forms.ModelChoiceField(queryset=Indicator.objects.all(),required=True,widget=forms.Select(attrs={
        'class' : 'form-control'
    }))
    is_actual = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={
        'class' : 'form-check'
    }))
    type = forms.CharField(required=True,widget=forms.Select(attrs={
        'class' : 'form-control'
    }))
    value = forms.CharField(required=True, widget=forms.TextInput(attrs={
        'class' : 'form-control'
    }))
    source = forms.ModelChoiceField(required=False,queryset=Source.objects.all(),widget=forms.Select(attrs={
        'class' : 'form-select mt-2'
    }))

#Value
class ValueForm(forms.ModelForm):
    value = forms.FloatField(required=True,widget=forms.Select(attrs={
        'class' : 'form-control'
    }))
    class Meta:
        model = DataValue
        fields = ('value',)
        
        widgets = {
            'value' : forms.TextInput(attrs={
                'class' : 'form-control'
            })
        }

class ValueForm2(forms.Form):
    value2 = forms.FloatField(required=True,widget=forms.Select(attrs={
        'class' : 'form-control'
    }))







############### DashBoard Topic ################3
class DashboardTopicForm(forms.ModelForm):
    class Meta:
        model = DashboardTopic
        fields = ('title_ENG', 'title_AMH', 'icon')
        widgets = {
            'title_ENG': forms.TextInput(attrs={'class': 'form-control'}),
            'title_AMH': forms.TextInput(attrs={'class': 'form-control'}),
        }
        

from ckeditor.widgets import CKEditorWidget


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ['title_ENG', 'title_AMH', 'for_catgory', 'content', 'is_dashboard_visible']
        widgets = {
            'title_ENG': forms.TextInput(attrs={'class': 'form-control'}),
            'title_AMH': forms.TextInput(attrs={'class': 'form-control'}),
            'for_catgory': forms.Select(attrs={'class': 'form-control'}),
            'content': CKEditorWidget(attrs={'class': 'form-control'}),
            'is_dashboard_visible': forms.CheckboxInput(attrs={'class': 'form-check-input ml-3'}),
        }

        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['for_catgory'].queryset = Category.objects.all()       



class CategoryProjectForm(forms.ModelForm):
    class Meta:
        model = Project 
        fields = ['title_ENG', 'title_AMH', 'for_catgory', 'content', 'is_dashboard_visible']
        widgets = {
            'title_ENG': forms.TextInput(attrs={'class': 'form-control'}),
            'title_AMH': forms.TextInput(attrs={'class': 'form-control'}),
            'for_catgory': forms.HiddenInput(),
            'content': CKEditorWidget(attrs={'class': 'form-control'}),
            'is_dashboard_visible': forms.CheckboxInput(attrs={'class': 'form-check-input ml-3'}),
        }



class CategoryVariableForm(forms.ModelForm):
    class Meta:
        model = Variables
        fields = ['title_ENG', 'title_AMH', 'for_catgory', 'content', 'is_dashboard_visible']
        widgets = {
            'title_ENG': forms.TextInput(attrs={'class': 'form-control'}),
            'title_AMH': forms.TextInput(attrs={'class': 'form-control'}),
            'for_catgory': forms.HiddenInput(),
            'content': CKEditorWidget(attrs={'class': 'form-control'}),
            'is_dashboard_visible': forms.CheckboxInput(attrs={'class': 'form-check-input ml-3'}),
        }

class VariableForm(forms.ModelForm):
    class Meta:
        model = Variables
        fields = ['title_ENG', 'title_AMH', 'for_catgory', 'content', 'is_dashboard_visible']
        widgets = {
            'title_ENG': forms.TextInput(attrs={'class': 'form-control'}),
            'title_AMH': forms.TextInput(attrs={'class': 'form-control'}),
            'for_catgory': forms.Select(attrs={'class': 'form-control'}),
            'content': CKEditorWidget(attrs={'class': 'form-control'}),
            'is_dashboard_visible': forms.CheckboxInput(attrs={'class': 'form-check-input ml-3'}),
        }

        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['for_catgory'].queryset = Category.objects.all()         