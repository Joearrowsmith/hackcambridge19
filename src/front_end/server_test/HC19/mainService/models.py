from django.db import models
import datetime
# Create your models here.

"""
This is a badly designed database, designed as a prototype for a single user.
If continued, then please change it. 
"""


"""
Metrics:
time, anger, sadness, happiness, fear, disgust, stress, productivity, journal note

Calender:
duration, text input, daily frequency, creation time, special

Health:
entry time, weight, standing hours, active energy resting energy steps, exercise minutes, flights climbed, walking heart rate, resting heart rate, heart rate variability, sleep, estimated sleep
"""

class Metrics(models.Model):
    entry_time = models.DateTimeField()
    anger = models.FloatField()
    sadness = models.FloatField()
    happiness = models.FloatField()
    fear = models.FloatField()
    stress = models.FloatField()
    productivity = models.FloatField()
    jounal_note = models.TextField()

    def add(vals):
        entry = Metrics.objects.create(
            entry_time = datetime.datetime.utcnow().date(),
            anger = vals['anger'],
            sadness = vals['sadness'],
            happiness = vals['happiness'],
            fear = vals['fear'],
            stress = vals['stress'],
            productivity = vals['productivity'],
            jounal_note = vals['jounal_note']
        )
        entry.save()

class GraphOutput(models.Model):
    date = models.DateField()
    kpi = models.FloatField()
    journal = models.TextField(max_length=500)

    def populate():
        print('populate called')
        current_date = datetime.datetime.utcnow().date()
        latest = Metrics.objects.filter(entry_time = current_date)[-1]
        print(latest)

class Calender(models.Model):
    duration = models.DurationField()
    text_input = models.TextField()
    daily_frequency = models.FloatField()
    creation_time = models.DateTimeField()
    special = models.TextField()

class Health(models.Model):
    entry_time = models.DateTimeField()
    weight = models.FloatField()
    standing_hours = models.FloatField()
    active_energy = models.FloatField()
    resting_energy = models.FloatField()
    exercise_minutes = models.FloatField()
    flights_climbed = models.FloatField()
    walking_heart_rate = models.FloatField()
    resting_heart_rate = models.FloatField()
    heart_rate_variability = models.FloatField()
    sleep = models.FloatField()
    estimated_sleep = models.FloatField()