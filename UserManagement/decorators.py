from django.http import HttpResponse
from django.shortcuts import redirect
from django.contrib import messages



def admin_user_required(view_func):
    def wrapper_func(request, *args, **kwargs):
        # Check if the user is authenticated
        if not request.user.is_superuser:
            # Redirect to the login page if the user is not authenticated
            return redirect('login')  # Make sure 'login' is the name of your login URL pattern

        # Check if the user has specific attributes
        if request.user.is_first_time:
            return redirect('user-admin-change-password')  # Adjust the URL name as needed

        # Check if the user has admin permissions
        if request.user.is_superuser and request.user.is_staff and request.user.is_active and not request.user.is_paid:
            return view_func(request, *args, **kwargs)
        else:
            # Redirect to a different page or show an error message
            return HttpResponse('You are not authorized to access this content.')
    
    return wrapper_func


def staff_user_required(view_func):
    def wrapper_func(request, *args, **kwargs):
        if request.user.is_first_time: 
            return  redirect('staff-change-password')
        elif request.user.is_staff and request.user.is_active:
            return view_func(request, *args, **kwargs)
        else:
             return HttpResponse('You are not authorized to access this content.')
    return wrapper_func


def dashboard_user_required(view_func):
    def wrapper_func(request, *args, **kwargs):
        if request.user.is_dashboard and request.user.is_active:          
            return view_func(request, *args, **kwargs)
        else:
             messages.error(request, 'You are not authorized to access this content.')
             return redirect('dashboard-login')
    return wrapper_func