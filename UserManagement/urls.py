from django.conf.urls.static import static
from django.conf import settings
from . import urls

from django import views
from django.urls import path
from UserManagement import views
from .views import *
from django.contrib.auth import views as auth_views


urlpatterns = [
    #Profile  
    path('admin-profile/', views.admin_profile, name="user-admin-profile"),

    #Password 
    path('admin-setting/', views.admin_change_password, name='user-admin-change-password'),
    path('staff-setting/', views.staff_change_password,  name='staff-change-password'),
    
    #User Operation 
    path('admin-update-profile/', views.admin_profile_updated, name='admin-update-profile'),
    path('staff-update-profile/', views.staff_profile_updated, name='staff-update-profile'),


    path('users/', views.users_list, name="user-admin-user-list"),
    path('activate_deactivate_user/<int:user_id>/', views.activate_deactivate_user, name='activate_deactivate_user'),
    path('users/<int:user_id>/edit/', views.edit_user, name='edit_user'),   

    path('forget-password/', forget_password, name='forget_password'),
    path('reset-password/<str:token>/<str:uidb64>/', reset_password, name='reset_password'),  # Updated URL pattern
    path('reset-password-confirm/', reset_password_confirm, name='reset_password_confirm'),
]
     