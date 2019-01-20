import csv
from requests_oauthlib import OAuth2Session

graph_url = 'https://graph.microsoft.com/v1.0'

def get_entry_info(event):
    entry = {'start':event['start']['dateTime'],
             'end':event['end']['dateTime'],
             'subject':event['subject'],
             'isReminderOn':event['isReminderOn'],
             'importance':event['importance'],
             'isAllDay':event['isAllDay'],
             'recurrence':event['recurrence']}
    return entry
  
def write_entry(lines, file, write_state='a'):
  with open(file, write_state, encoding='utf-8') as f:
    wtr = csv.writer(f, delimiter=',')
    wtr.writerow(lines)


def get_next_n_steps(token, n_steps=3, step_size=2):
    count = 0
    for i in range(n_steps):
      events = get_calendar_events(token, 
                                   step=step_size, 
                                   skip=step_size*i)
      if events:
        for idx, event in enumerate(events['value']):
          count += 1
          entry = get_entry_info(event)
          print(idx, count, entry['subject'])
          

def get_all_events(token, step_size=100):
  count = 0
  loop = True
  write_entry(['start','end','subject'
               ,'isReminderOn','importance','isAllDay',
               'recurrence'], 'data.csv', write_state='w')
  
  while loop:
    events = get_calendar_events(token, 
                                 step=step_size, 
                                 skip=count)

    if len(events['value']) > 0:
      values = []
      for idx, event in enumerate(events['value']):
        count += 1
        entry = get_entry_info(event)
        line = [entry['start'],
                entry['end'],
                entry['subject'],
                entry['isReminderOn'],
                entry['importance'],
                entry['isAllDay'],
                entry['recurrence']]
        write_entry(line, 'data.csv', 'a')
    else:
      loop = False
      break
      
      
def calendar(request):
  context = initialize_context(request)
  token = get_token(request)
  get_all_events(token)
  
  
def get_calendar_events(token, step=10, skip=0):
  graph_client = OAuth2Session(token=token)
  query_params = {
    '$select': 'subject,isReminderOn,start,end,importance,isAllDay,recurrence',
    '$orderby': 'start/DateTime DESC',
    '$top' : step,
    '$skip' : skip
  }

  events = graph_client.get('{0}/me/events'.format(graph_url), 
                            params=query_params)
  
  return events.json()
