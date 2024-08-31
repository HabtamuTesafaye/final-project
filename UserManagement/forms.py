from django import forms
from .models import CustomUser
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.views import PasswordChangeView
from django.urls import reverse_lazy
from django.shortcuts import render


class PasswordChangingForm(PasswordChangeView):
    form_class = PasswordChangeForm
    template_name = 'user-admin/setting.html'
    success_url = reverse_lazy('your_success_url')


class CustomUserForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'first_name', 'last_name', 'is_active']  

        widgets = {
            'password': forms.PasswordInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
        }
         

class CustomUserForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'first_name','last_name', 'photo', 'is_superuser']   

        widgets = {
                'password': forms.PasswordInput(attrs={'class': 'form-control'}),
                'photo' : forms.ClearableFileInput(attrs={
                    'class' : 'form-control'
                })
            } 

class EditProfileForm(forms.ModelForm):
    error_css_class = 'text-danger'
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'first_name','last_name', 'photo']   

        widgets = {
            'username' : forms.TextInput(attrs={'class': 'form-control'}),
            'email' : forms.EmailInput(attrs={'class': 'form-control'}),
            'first_name' : forms.TextInput(attrs={'class': 'form-control'}),
            'last_name' : forms.TextInput(attrs={'class': 'form-control'}),
                'password': forms.PasswordInput(attrs={'class': 'form-control'}),
                'photo' : forms.ClearableFileInput(attrs={
                    'class' : 'form-control'
                })
            } 
        

class Login_Form(forms.Form):

    email = forms.EmailField(label='Email', widget=forms.EmailInput(attrs={
        'class' : 'form-control',
        'placeholder' : 'Enter Your Email'
    }))
    password = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={
        'class' : 'form-control',
        'placeholder' : 'Enter Your Password'
    }))
    class Meta:
        fields = ['email', 'password']


class LoginFormDashboard(forms.Form):

    email = forms.EmailField(label='',widget=forms.EmailInput(attrs={
        'class' : 'form-control',
        'placeholder' : 'Email address'
    }))
    password = forms.CharField(label='',widget=forms.PasswordInput(attrs={
        'class' : 'form-control',
        'placeholder' : 'Password'
    }))


class CustomUserCreationForm(forms.Form):
    first_name = forms.CharField(max_length=30, widget=forms.TextInput(attrs={
        'class' : 'form-control',
        'placeholder' : 'Your First Name'
    }))
    last_name = forms.CharField(max_length=30, widget=forms.TextInput(attrs={
        'class' : 'form-control',
        'placeholder' : 'Your Last Name'
    }))
    username = forms.CharField(max_length=30, widget=forms.TextInput(attrs={
        'class' : 'form-control',
        'placeholder' : 'Enter your Username',
        'autocomplete': 'off' 
    }))
    email = forms.EmailField(widget=forms.EmailInput(attrs={
        'class' : 'form-control',
        'placeholder' : 'Enter your Email',
        'autocomplete': 'off'
    }))
    # password1 = forms.CharField( max_length=40, label='Password' ,widget=forms.PasswordInput(attrs={
    #     'class' : 'form-control',
    #     'placeholder' : 'Enter Your Password',
    #     'autocomplete': 'off'
    # }))
    # password2 = forms.CharField( max_length=40, label='Confirm Password', widget=forms.PasswordInput(attrs={
    #     'class' : 'form-control',
    #     'placeholder' : 'Confirm Password',
    #     'autocomplete': 'off'
    # }))
    photo = forms.ImageField(required=False, widget=forms.ClearableFileInput(attrs={
        'class' : 'form-control',
        'placeholder' : 'Add Photo(Optional)',
   
    }))


    is_superuser = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={
        'class' : 'form-check-input'
    }))



