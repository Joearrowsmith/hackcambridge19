import pandas as pd
import re

def clean_data(text):
    text = str(text).replace(' ', '_')
    text = re.sub(r'\W+', '', text)
    text = text.lower()
    text = str(text).replace('_', ' ')
    text = re.sub(' +', ' ', text).strip()
    return text

if __name__=='__main__':
    df = pd.read_csv('data/calendar_data.csv')

    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
  
    df['subject'] = df['subject'].apply(clean_data)
  
    print(df['subject'])