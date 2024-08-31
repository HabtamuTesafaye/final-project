from django import forms
# from UserManagement.models import CustomUser
from django.contrib.auth.forms import PasswordChangeForm,UserChangeForm
from django.contrib.auth.models import User  
from UserManagement.models import CustomUser
from django.contrib.auth.forms import SetPasswordForm
from .models import DashboardTopic , Project , Category



class CustomUserSetPasswordForm(SetPasswordForm):
    new_password1 = forms.CharField(
        label='New Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'New Password',
        })
    )
    new_password2 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm Password',
        })
    )

    class Meta:
        model = CustomUser
        fields = ['new_password1', 'new_password2']


        
class PasswordChangingForm(PasswordChangeForm):
    old_password=forms.CharField(widget= forms.PasswordInput(attrs={
                'class' : 'form-control','placeholder': 'old password'}))
    new_password1=forms.CharField(widget= forms.PasswordInput(attrs={
                'class' : 'form-control','placeholder': 'new password'}))
    new_password2=forms.CharField(widget= forms.PasswordInput(attrs={
                'class' : 'form-control','placeholder': 'confirm password'}))
    class Meta:
        model=CustomUser
        fields=['old_password','new_password1','new_password2']
        
class UserChangingForm(UserChangeForm):
    username=forms.CharField(widget= forms.TextInput(attrs={
                'class' : 'form-control','placeholder': 'username'}))
    last_name=forms.CharField(widget= forms.TextInput(attrs={
                'class' : 'form-control','placeholder': 'last_name'}))
    first_name=forms.CharField(widget= forms.TextInput(attrs={
                'class' : 'form-control','placeholder': 'first_name'}))
    email=forms.CharField(widget= forms.EmailInput(attrs={
                'class' : 'form-control','placeholder': 'email'}))
    class Meta:
        model=CustomUser
        fields= ['username','last_name','first_name','email'] 
# class profilepcform(forms.ModelForm):
#     profile_image=forms.ImageField(label="profile picture")  
#     class Meta:
#         model=CustomUser     
#         fields=('image',)
        
    from django import forms

class Login_Form(forms.Form):
    username = forms.CharField(max_length=150, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter your username'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Enter your password'}))






##REDACTOR_wysiwyg_EDITOR






    