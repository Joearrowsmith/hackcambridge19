# journAI space
## A data-driven approach to mental health

### Hack Cambridge 2019 Submission

Team members:
- Joonsu Gha
- Edward Stables
- Hu Fangfang
- Joe Arrowsmith

## User Experience

![Dashboard](misc/bashboard1.png)

![Dashboard](misc/kpi-journal-entry-graph.png)

![User Profile](misc/profile1.png)

![User Profile](misc/profile2.png)

![Treatment](misc/treatment.png)

![Help](misc/help.png)

![Password](misc/bashboard1.png)

## Backend Framework:

The backend was programmed using the django framework. Django serves the user a preset page template depending on their url, as well as making database interactions quick and easy. The database was setup to contain the information viewed by the webapp user, as well as the information needed for the machine learning models. Due to some issues with getting data from the user, there wasn't time to fully integrate the UI with the database information, but a small amount more work to integrate these will give the webapp full functionality.

---

## Prediction Models:

![LSTM performance](misc/LSTM_performance.png)

![LSTM ensemble](misc/LSTM_performance_ensemble.png)

![Confusion Matrix](misc/confusion_matrix_spacy.png)
