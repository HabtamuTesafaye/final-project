from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib import messages
from django.urls import reverse
from TimeSeriesBase.models import *
from UserAdmin.forms import *
from .forms import *
from django.shortcuts import render, redirect
from .models import CustomUser
from django.contrib.auth import login,authenticate,logout,get_user_model
from  .decorators import admin_user_required, staff_user_required
from django.contrib.auth.decorators import login_required
import random
import string
from django.core.mail import EmailMultiAlternatives
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import update_session_auth_hash
from TimeSeriesBase.forms import CustomUserSetPasswordForm
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes, smart_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils import timezone
from django.urls import reverse_lazy
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
import threading
from django.template.loader import render_to_string

from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string

from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string

def send_registration_email(email, first_name, last_name, auto_password):
    subject = "Account Creation"
    from_email = "beteltesfu15@gmail.com"
    to = email
    text_content = "Account Created Successfully"
    
    context = {
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'password': auto_password,
    }
    html_content = render_to_string('success-email.html', context)
    
    msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
    msg.attach_alternative(html_content, "text/html")
    
    try:
        print(f"Sending registration email to {email}...")
        result = msg.send()
        if result == 1:
            print(f"Email sent successfully to {email}.")
        else:
            print(f"Failed to send email to {email}.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def send_reset_email(user, reset_url):
    subject = "Reset Password"
    from_email = "beteltesfu15@gmail.com"
    to = user.email
    text_content = "Account Password Reset Successfully"
    
    context = {
        'first_name': user.first_name,
        'last_name': user.last_name,
        'email': user.email,
        'reset_url': reset_url,
    }
    html_content = render_to_string('reset_password_email.html', context)
    
    msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
    msg.attach_alternative(html_content, "text/html")
    
    try:
        print(f"Sending reset email to {user.email}...")
        result = msg.send()
        if result == 1:
            print(f"Email sent successfully to {user.email}.")
        else:
            print(f"Failed to send email to {user.email}.")
    except Exception as e:
        print(f"Failed to send email: {e}")


User = get_user_model()

import threading

def forget_password(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            validate_email(email)  # Validate email format
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            messages.error(request, 'User with this email does not exist.')
            return redirect('forget_password')
        except ValidationError:
            messages.error(request, 'Invalid email format.')
            return redirect('forget_password')
        
        try:
            # Generate token and uidb64
            token = default_token_generator.make_token(user)
            uidb64 = urlsafe_base64_encode(force_bytes(user.pk))

            # Generate reset URL
            reset_url = reverse_lazy('reset_password', kwargs={'token': token, 'uidb64': uidb64})
            reset_url = request.build_absolute_uri(reset_url)

            # Start a background thread to send the reset email
            background_thread = threading.Thread(target=send_reset_email, args=(user, reset_url))
            background_thread.start()

            messages.success(request, 'Password reset link has been sent to your email.')
            return redirect('forget_password')
        except Exception as e:
            # Handle network-related errors
            messages.error(request, 'An error occurred while sending the reset password link. Please try again later.')
            print(f"Error: {e}")

    return render(request, 'forget_pass.html')

# views.py
def reset_password(request, token, uidb64):
    try:
        uid = smart_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        # Check if last_reset_password is set and compare the time
        if user.last_reset_password and timezone.now() - user.last_reset_password > timezone.timedelta(hours=1):
            messages.error(request, 'The reset password link has expired. Please request a new one.')
            return redirect('forget_password')

        if request.method == 'POST':
            form = CustomUserSetPasswordForm(user, request.POST)
            if form.is_valid():
                password = form.cleaned_data['new_password1']
                form.save()

                # Log the user in after password reset
                user = authenticate(request, username=user.email, password=password)
                login(request, user)

                # Redirect to the success page
                messages.success(request, 'Password reset successfully. You are now logged in.')
                return redirect('reset_password_confirm')
            # Do not add the generic error message here
        else:
            form = CustomUserSetPasswordForm(user)

        return render(request, 'reset_password.html', {'form': form})
    else:
        messages.error(request, 'The reset password link is invalid or has expired. Please request a new one.')
        return redirect('forget_password')
  
def reset_password_confirm(request):
    return render(request, 'sucess.html')

#forget password end
def generate_password(length):
    characters = string.ascii_letters + string.digits 
    password = ''.join(random.choice(characters) for _ in range(length))
    return password

@login_required(login_url='login')
@admin_user_required
def users_list(request):
    site_config = get_object_or_404(SiteConfiguration, id=1)  # Adjust the lookup based on your model logic

    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        form1 = SiteConfigurationForm(request.POST, instance=site_config)

        if form.is_valid():
            # Your user creation logic here...
            username = form.cleaned_data['username']
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']
            email = form.cleaned_data['email']
            is_admin = form.cleaned_data['is_superuser']
            
            auto_password = generate_password(10)
            user_obj = CustomUser.objects.create_user(
                email=email,
                first_name=first_name,
                is_superuser=is_admin,
                last_name=last_name,
                is_active=True,
                is_staff=True,
                username=username,
                password=auto_password
            )
            
            # Start the background thread without extra arguments
            background_thread = threading.Thread(
                target=send_registration_email,
                args=(email, first_name, last_name, auto_password),  # Only pass the required arguments
                daemon=True
            )
            
            # Start the background thread
            background_thread.start()

            messages.success(request, "Your Account has been Successfully Created! You will receive an email.")
            return redirect('user-admin-user-list')



        elif form1.is_valid():
            form1.save()
            messages.success(request, 'Site configuration updated successfully.')
            return redirect('user-admin-user-list')

        else:
            messages.error(request, 'Please correct the errors in the form.')

    else:
        form = CustomUserCreationForm()
        form1 = SiteConfigurationForm(instance=site_config)

    total_users_count = CustomUser.objects.count()
    active_users_count = CustomUser.objects.filter(is_active=True).count()
    inactive_users_count = total_users_count - active_users_count
    users = CustomUser.objects.all()

    context = {
        'form': form,
        'form1': form1,
        'users': users,
        'total_users_count': total_users_count,
        'active_users_count': active_users_count,
        'inactive_users_count': inactive_users_count,
    }

    return render(request, 'user-admin/users_list.html', context)

@login_required(login_url='login')
def logout_view(request):
    logout(request)
    return redirect('login')


def login_view(request):
    if request.method == 'POST':
        form = Login_Form(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request,email=email,password=password)

            if user is not None and user.is_superuser:
                login(request, user)
                return redirect('user-admin-index')
            elif user is not None and user.is_staff:
                login(request, user)
                return redirect('index')
            else:
                messages.error(request, 'Invalid Password or Email')
            form = Login_Form()
    else:
        form = Login_Form()
    return render(request, 'login.html', {'form': form})

@login_required(login_url='login')
@admin_user_required
def edit_user(request, user_id):
        user =  CustomUser.objects.get(id = user_id)
        if request.method == 'POST':
            form = CustomUserForm(request.POST, instance=user)
            if form.is_valid():
                form.save()
                messages.success(request, "User updated successfully!")
                return redirect('user-admin-user-list')  # Change 'user-admin-user-list' to the appropriate URL name or path
        else:
            form = CustomUserForm(instance=user)
    
        return render(request, 'user-admin/edit_user.html', {
            'form': form,
            'user': user,
        })

@login_required(login_url='login')
@admin_user_required
def admin_profile(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            form = CustomUserCreationForm()   
    else:
        form = CustomUserCreationForm()

    return render(request, 'user-admin/profile.html', {'form': form})


@login_required(login_url='login')
@admin_user_required
def admin_profile_updated(request):
    user = CustomUser.objects.get(pk = request.user.pk)
    form = EditProfileForm(request.POST or None, request.FILES or None,instance=user)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully Updated!')
        else:
            messages.error(request, 'Please tye again!')         
    return render(request, 'user-admin/profile.html', {'form': form, 'user': user})


@login_required(login_url='login')
@staff_user_required
@admin_user_required
def staff_profile_updated(request):
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
    return render(request, 'setting.html', context)


@login_required(login_url='login')
@admin_user_required
def activate_deactivate_user(request, user_id):
    try:
        user = CustomUser.objects.get(pk = user_id)
        previous_page = request.META.get('HTTP_REFERER')
        
        if request.method == 'POST':
            user.is_active = not user.is_active  # Toggle the is_active status
            user.save()
            messages.success(request, f"User '{user.first_name} {user.last_name}' has been {'Activated ' if user.is_active else 'Deactivated'}!")
            return HttpResponseRedirect(previous_page)
    except:
        messages.error(request, "Please Try Again Later!")

    return render(request, 'user-admin/users_list.html', {'user': user})


@login_required(login_url='login')
def admin_change_password(request):
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
    
    return render(request,'user-admin/setting.html', {'formChangePassword': form})


@login_required(login_url='login')
def staff_change_password(request):
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
    
    return render(request,'password.html', {'form': form})




 