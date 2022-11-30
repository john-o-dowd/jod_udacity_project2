# Disaster Response Pipeline Project

## Summary:
In the event of a disaster large numbers of messages are received from the public and emergencey services. The ability to triage these messages and direct urgent messages to the relavant authority in a timely manner is critical to dealing with a disaster efficently. This code takes prelabled messages from previous disasters which have are used to train a model. This model is then used to label new messages in ongoing disaster situations which can then be used to pass the messages on to relevant autorities with the appropriate level of urgency.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. App runs by default on localhost:3100


## Initial landing page fives statitics about the training dataset:

##### Breaddown of messages based on genre:
![Alt text](/readme_images/DistributionOfMessageGenres.png?raw=true "Title")

#### Breadown of messages based on presence of weather in message: 
![Alt text](/readme_images/WeatherRelatedMessages.png?raw=true "Title")

## Files in repository:
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- modelTest.py # script to analyse model perfomance
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- disaster_categories_simples.csv # subset of data to process for testing/debug 
|- disaster_messages_simples.csv # subset of data to process for testing/debug
|- process_data.py # load and clean csv data before save to sql database
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py # Build and save classifer model to categorize disaster messages. The model is trained based on user
    supplied pre-categorized training messages
|- classifier.pkl # saved model
readme_images #images for readme
README.md
