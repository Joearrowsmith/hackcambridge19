from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('tester', views.tester, name='tester'),
    path('signin', views.sign_in, name='signin'),
    path('callback', views.callback, name='callback'),
    path('signout', views.sign_out, name='signout'),
    path('calendar', views.calendar, name='calendar'),
]