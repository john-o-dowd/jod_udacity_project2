# Disaster Response Pipeline Project

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



