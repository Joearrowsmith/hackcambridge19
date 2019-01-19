from django.db import models

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
    disgust = models.FloatField()
    stress = models.FloatField()
    productivity = models.FloatField()
    jounal_note = models.TextField()

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