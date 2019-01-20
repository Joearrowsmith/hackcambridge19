# journAI space
## A data-driven approach to mental health

### Hack Cambridge 2019 Submission

Team members:
- Joonsu Gha
- Edward Stables
- Hu Fangfang
- Joe Arrowsmith

## User Experience

The frontend was built upon an MIT licensed bootstrap template, whereby a calendar, journal entry, mood chart, emergency email and hide current screen functionalities were implemented. The overall layout and design of these components were formulated with the user's privacy as one of the most important aspects, due to the sensitive nature of mental health. For instance, the initial dashboard is reminiscent of the usual productivity application with the mood chart accessible by scrolling down.  The hide screen feature was also incorporated for cases where the user feels the need to keep his dashboard hidden from those around him, until it is unlocked by the user himself with a password (we did not manage to implement the password unlocking feature). The email functionality was designed with the idea of having an auto filled message body which would have been set by the user beforehand for use in cases whereby the user wants to contact their emergency contact persons but may not feel well enough to be able to type out that they are unwell. The ongoing treatments page allows the user to have a quick checklist of their necessary medications or appointments, facilitating their healing process. In the user profile, the user can set a personal message to themselves which will be shown whenever they are on that page.

![Dashboard](misc/dashboard1.png)

![Journal Graph](misc/kpi-journal-entry-graph.png)

![User Profile](misc/profile1.png)

![User Profile](misc/profile2.png)

![Treatment](misc/treatment.png)

![Help](misc/help.png)

![Password](misc/password.png)

## Backend Framework:

The backend was programmed using the django framework. Django serves the user a preset page template depending on their url, as well as making database interactions quick and easy. The database was setup to contain the information viewed by the webapp user, as well as the information needed for the machine learning models. Due to some issues with getting data from the user, there wasn't time to fully integrate the UI with the database information, but a small amount more work to integrate these will give the webapp full functionality.

---

## Prediction Models:

A deep-learning based Natural Language Processing model was built for the automatic recognition of emotion from the journal entry of the user. The emotion variables, along with the user's calendar and health data, are used to calculate the Key Performance Index (KPI).

![LSTM performance](misc/LSTM_performance.png)

![LSTM ensemble](misc/LSTM_performance_ensemble.png)

![Confusion Matrix](misc/confusion_matrix_spacy.png)

## Results



## Moving Forward