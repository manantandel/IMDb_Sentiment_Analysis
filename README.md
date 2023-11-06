# IMDb Sentiment Analysis
![](https://i.imgur.com/q5IiPXi.png)
## Streamlit Deployment Link
[IMDb Sentiment Analysis](https://imdb-sentiment-analysis.streamlit.app/)
## Project Details
### Introduction
IMDb is an online database of information related to entertainment industry, it contains various data related to movies, television series, podcasts and several other form of media.
### Objective
The objective of this project is to develop a sentiment analysis model for IMDb movie reviews. The problem is to analyze and classify a review or set of reviews into positive or negative while also considering the star rating.
### Dataset Description
- The dataset was created by scraping data using Selenium and the IMDb ID of movies.
- Review title, star rating and actual review were scraped for each review of the movie.
- The final size of dataset was 111555 independent entries of reviews and respective user rating.
### Python Libraries Used
```
Streamlit
Beautiful Soup 
Pandas
Plotly
Requests 
Scikit-learn
```
### Models trained
1. KNN Classifier Algorithm
   - Accuracy: 73%
2. Multinomial Naïve-Bayes Algorithm
   - Accuracy: 76%
3. Logistic Regression Model
   - Accuracy: 87%
4. Decision Tree
   - Accuracy: 75%
