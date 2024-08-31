from .views import (topic_lists, index , payment  ,update_unique_id , payment_callback , return_type ,payment_history,website_trained_list ,dashboard_profile,change_password,profile_updated,serve_json_data, payment_callback,dashboard_register, dashboard_logout, category_list , category_detail_lists , pie_chart_data, search_category_indicator, dashboard_login,chat_view,website_info_view,learning_data_view,chatbot)

from .views import (topic_lists, index  , category_list , category_detail_lists, indicator_detail,pie_chart_data)
from django.urls import path
from django.contrib.auth import views as auth_views
from .forms import UserPasswordResetForm, UserPasswordConfirmForm


urlpatterns = [
    path('',index, name="dashboard-index"),
    path('register/', dashboard_register, name='dashboard_register'),
    path('login/',dashboard_login, name="dashboard-login"),
    path('logout/',dashboard_logout, name="dashboard-logout"),
    path('topic_lists/',topic_lists, name="topic_lists"),
    path('pie_chart_data/',pie_chart_data, name="pie_chart_data"),
    path('category_list/<int:id>/',category_list, name="category_list"),
    path('category_detail_list/<int:id>/',category_detail_lists, name="category_detail_lists"),
    path('indicator-detail/<int:id>/',indicator_detail, name='indicator-detail'),
    path('search_category_indicator/',search_category_indicator, name='search_category_indicator'),

    ##### payment 
    path('payment/', payment, name='payment'),
    path('cb/', return_type, name="return_type"),
    path('callback/', payment_callback, name="callback"),
    path('update-unique-id/', update_unique_id, name='update_unique_id'),
    path('payment_history/', payment_history, name='payment_history'),
    path('dashboard_profile',dashboard_profile , name='dashboard_profile') , 
    path('change_password',change_password , name='change_password') , 
    path('update_profile',profile_updated , name='update_profile') , 
    path('json-data/', serve_json_data, name='serve-json-data'),


    #### RESET PASSWORD
    #dashboard-pages/authentication/
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='dashboard-pages/authentication/reset_password.html', form_class=UserPasswordResetForm), name='password_reset'),
    path('password_reset_done/', auth_views.PasswordResetDoneView.as_view(template_name='dashboard-pages/authentication/password_reset_done.html'), name='password_reset_done'),
    path(r'reset/<uidb64>/<token>/',auth_views.PasswordResetConfirmView.as_view(template_name="dashboard-pages/authentication/password_reset_confirm.html",form_class=UserPasswordConfirmForm), name='password_reset_confirm'),
    path('password_reset_complete/', auth_views.PasswordResetCompleteView.as_view(template_name="dashboard-pages/authentication/password_reset_complete.html"), name='password_reset_complete'),


    path("chat", chatbot, name='home'),
    path('chat_view/', chat_view, name='chat_view'),
    path('website-info/', website_info_view, name='website_info'),
    path('learning-data/', learning_data_view, name='learning_data'),

     path('website-train/', website_trained_list, name='website_train_list'),
    

]
