from django import forms

class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)
    your_age = forms.DecimalField(label = 'your age', max_value= 150)