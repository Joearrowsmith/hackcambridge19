from django import forms

class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)
    your_age = forms.DecimalField(label = 'Your age', max_value= 150)

class JournalForm(forms.Form):

    entry = forms.CharField(label='Entry', max_length=500, widget = forms.Textarea(attrs={'cols': 80, 'rows': 4}))
    happiness = forms.DecimalField(label='Happiness', max_value=100, min_value=0)
    sadness = forms.DecimalField(label='Sadness', max_value=100, min_value=0)
    anger = forms.DecimalField(label='Anger', max_value=100, min_value=0)
    disgust = forms.DecimalField(label='Disgust', max_value=100, min_value=0)
    productivity = forms.DecimalField(label='Productivity', max_value=100, min_value=0)
    stress = forms.DecimalField(label='Stress', max_value=100, min_value=0)
