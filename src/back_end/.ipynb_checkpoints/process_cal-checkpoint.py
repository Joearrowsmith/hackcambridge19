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
                history.append(state)
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
    df = pd.read_csv('data/calendar_data.csv')

    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
  
    df['subject'] = df['subject'].apply(clean_data)
    df['recurrence'] = df['recurrence'].apply(clean_recurrence)
   
    
    history = get_day_statuses(df)
    
    for i in history[-10:]:
        print(i)
        print('---')