from django import forms
from django.contrib.auth.forms import PasswordResetForm
from django.contrib.auth.forms import SetPasswordForm


class UserPasswordResetForm(PasswordResetForm):
    def __init__(self, *args, **kwargs):
        super(UserPasswordResetForm, self).__init__(*args, **kwargs)

    email = forms.EmailField(label='',widget=forms.EmailInput(attrs={
        'class': 'form-control',
        'placeholder' : 'Enter your email',
        }))


class UserPasswordConfirmForm(SetPasswordForm):
    new_password1 = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={
        'class': 'form-control w-100',
        'placeholder': 'Password',
    }))
    new_password2 = forms.CharField(label='Conform Password', widget=forms.PasswordInput(attrs={
        'class': 'form-control w-100',
        'placeholder': 'Confirm Password',
    }))
# payments/forms.py
from django import forms

class PaymentForm(forms.Form):
    email = forms.EmailField(required=True)
    amount = forms.DecimalField(required=True, min_value=0)
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)

# chatbot
from .models import WebsiteInfo

class WebsiteInfoForm(forms.ModelForm):
    class Meta:
        model = WebsiteInfo
        fields = ['key', 'value']

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField()