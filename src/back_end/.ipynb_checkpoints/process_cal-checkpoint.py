import pandas as pd
import re
import math
import json

def clean_data(text):
    text = str(text).replace(' ', '_')
    text = re.sub(r'\W+', '', text)
    text = text.lower()
    text = str(text).replace('_', ' ')
    text = re.sub(' +', ' ', text).strip()
    return text

def clean_recurrence(text):
    if type(text) is str:
        text = text.replace("'",'"')
        d = json.loads(text)
        text = d['pattern']['type']
    else:
        text = '0'
    return text

def get_day_statuses(df):
    day = None
    history = []
    state = {}
    for idx, row in df.iterrows():
        current = row['start'].date()
        if not day == None:
            if day != current:
                ## new day, save old state
                if state != {}:
                    history.append([state['date'], 
                                    state['first'],
                                    state['last'],
                                    state['duration'],
                                    state['text'],
                                    state['allDay'],
                                    state['count']])
                state = {}
                state['date'] = current
                state['first'] = row['start'].time()
                state['last'] = row['end'].time()
                dur = row['end'] - row['start']
                state['duration'] = dur.seconds // 3600
                state['text'] = row['subject']
                state['allDay'] = row['isAllDay']
                state['count'] = 1
            else:
                state['date'] = current
                if state['first'] > row['start'].time():
                    state['first'] = row['start'].time()
                if state['last'] < row['end'].time():
                    state['last'] = row['end'].time()
                dur = row['end'] - row['start']
                state['duration'] += dur.seconds / 3600
                state['text'] += " gap " + row['subject']
                state['allDay'] = row['isAllDay'] or state['allDay']
                state['count'] += 1
        day = current
    
    return history
    
if __name__=='__main__':
    df_cal = pd.read_csv('data/calendar_data.csv')
    df_cal['start'] = pd.to_datetime(df_cal['start'])
    df_cal['idx'] = df_cal['start']
    df_cal = df_cal.set_index('idx')
    df_cal['end'] = pd.to_datetime(df_cal['end'])
    df_cal['subject'] = df_cal['subject'].apply(clean_data)
    df_cal['recurrence'] = df_cal['recurrence'].apply(clean_recurrence)
   
    
    df_ener = pd.read_csv('data/ActiveEnergyBurned.csv')
    df_ener = df_ener.drop(['Unnamed: 0'], axis=1)
    df_ener['endDate'] = pd.to_datetime(df_ener['endDate'])
    df_ener = df_ener.set_index("endDate")
    df_ener = df_ener.resample('D').sum()
    
    history = get_day_statuses(df_cal)
    
    df_hist = pd.DataFrame(history, columns=['date','first','last','duration','text','allDay','count'])
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    df_hist = df_hist.set_index('date')
    df_hist = df_hist.join(df_ener)
    
    
    
    