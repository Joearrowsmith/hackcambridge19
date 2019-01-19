from django.urls import path

from . import views

urlpatterns = [
    path('test/', views.tester, name='tester'),
    path('signin/', views.sign_in, name='signin'),
    path('callback/', views.callback, name='callback'),
    path('signout/', views.sign_out, name='signout'),
]