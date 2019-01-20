from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from mainService.auth_helper import get_sign_in_url, get_token_from_code, store_token, store_user, remove_user_and_token, get_token
from mainService.graph_helper import get_user, get_calendar_events
import dateutil.parser

from .forms import NameForm, JournalForm
from .models import Metrics, GraphOutput
# Create your views here.

def user_page(request):
  return render(request, 'mainService/user.html')
def treatment_page(request):
  return render(request, 'mainService/treatment.html')
def help_page(request):
  return render(request, 'mainService/help.html')

def main_page(request):
  if request.method == 'POST':
    if 'metric_submission' in request.POST:
      print("Journal Entry Made:")
      data_list = {}
      for key, val in request.POST.items():  
        if key not in  ['csrfmiddlewaretoken', 'metric_submission']:
          data_list[key] = val
      print(data_list)
      Metrics.add(data_list)
  context = initialize_context(request)

  return render(request, 'mainService/index.html', context)


#rewrite home_page as a class once input needs to be taken
def tester(request):
  if request.method == 'POST':
    print('POST request')
    if 'metric_submission' in request.POST:
      print(request.POST)
  else:
    print('GET request')

  context = initialize_context(request)

  return render(request, 'mainService/formTest.html', context)    

"""
ALL VIEWS FOLLWING THIS ARE TAKEN FROM THE MICROSOFT TUTORIAL ON USING THEIR GRAPH API
"""

def initialize_context(request):
  context = {}

  # Check for any errors in the session
  error = request.session.pop('flash_error', None)

  if error != None:
    context['errors'] = []
    context['errors'].append(error)

  # Check for user in the session
  context['user'] = request.session.get('user', {'is_authenticated': False})
  return context

def sign_in(request):
  # Get the sign-in URL
  sign_in_url, state = get_sign_in_url()
  # Save the expected state so we can validate in the callback
  print('STATE: ' + state)
  request.session['auth_state'] = state
  # Redirect to the Azure sign-in page
  return HttpResponseRedirect(sign_in_url)

def sign_out(request):
  # Clear out the user and token
  remove_user_and_token(request)

  return HttpResponseRedirect(reverse('tester'))

def callback(request):
  # Get the state saved in session
  expected_state = request.session.pop('auth_state', '')
  # Make the token request
  token = get_token_from_code(request.get_full_path(), expected_state)

  # Get the user's profile
  user = get_user(token)

  # Save token and user
  store_token(request, token)
  store_user(request, user)

  return HttpResponseRedirect(reverse('tester'))