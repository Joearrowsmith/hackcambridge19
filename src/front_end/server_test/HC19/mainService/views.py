from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.


#rewrite home_page as a class once input needs to be taken

def home_page(request):
    #point this to the html file when the templates are done
    return HttpResponse("Testing main page")