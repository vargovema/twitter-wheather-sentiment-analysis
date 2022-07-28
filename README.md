# twitter-wheather-sentiment-analysis

Folder Structure:

/Twitter/twitterProducer.ipynb ---> Notebook with code for starting the stream.
/Twitter/KafkaConsumerData_and_SentimentAnalysis.ipynb ---> Notebook with code for saving the data from stream and running the sentiment analysis. 
/Twitter/results_tweets ---> Saved Tweets from the stream for each of 8 locations. (The outputs with tweets_city_1_location are the final ones, and the tweets_city_location are the testing ones whether the stream was running correctly)
-> Both twitterProducer and KafkaConsumerData_and_SentimentAnalysis need to run simultaneously.

/weather ---> Weather data for 01.07.2021 accessed from OpenWeather for each of 8 locations.
/tweets_with_weather ---> Final merged dataset (Twitter data + Weather data) for each of 8 locations. 

/weatherdata.ipynb ---> Notebook with code for accessing of Weather data.
/data_processing_and_analysis.ipynb ---> Notebook with code for combining datasets, models (Linear model, Logistic Regression, Naive Bayes) and Visualisations.
 
/APIKeys.txt ---> Twitter API keys
/ReadMe.txt ---> Folder structure
