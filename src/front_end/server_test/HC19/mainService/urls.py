from django.urls import path

from . import views

urlpatterns = [
    path('', views.main_page, name='main_page'),
    path('user/', views.user_page, name='user_page'),
    path('treatment/', views.treatment_page, name='treatment_page'),
    path('help/', views.help_page, name='help_page'),
    path('test/', views.tester, name='tester'),
    path('signin/', views.sign_in, name='signin'),
    path('callback/', views.callback, name='callback'),
    path('signout/', views.sign_out, name='signout'),
]