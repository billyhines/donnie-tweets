# donnie-tweets

An analysis of @realDonaldTrump tweet behavior with the goal of predicting total tweet numbers for the PredictIt market  

## Motivations

I recently read an article from The Ringer about a political betting market called PredictIt [(link)](https://www.theringer.com/2018/3/21/17130490/predictit-politics-elections-gambling). PredictIt is a market where traders can speculate on the outcome of future political events. One of the markets that caught my eye is a weekly market speculating the total number of tweets to come from the @realDonaldTrump account. I figured it would be easy enough to get my hands on the data and to see if I could beat this market. This repo contains both a Jupyter Notebook and a Plotly Dash dashboard for visualizing and making predictions for this market.

## Instructions

The Jupyter Notebook uses Plotly for some of the plots, as I knew I wanted to build a dashboard later on using Plotly's Dash. The dashboard can be run by calling `python Dashboard.py`   

The Jupyter Notebook shows the entire process of starting with the tweets through making predictions for the current market. To pull the tweets, I've been using the [Trump Twitter Archive](http://www.trumptwitterarchive.com/) to get a .csv output of all of the tweets from the @realDonaldTrump account from his inauguration date. After some cleaning, I hit the PredictIt API to get the current market prices. The prediction part relies solely on previous tweet counts from the account. At a given hour _x_, I look at how many tweets the account has tweeted after that hour _x_ in the past. With the help of the standard deviation and skew of these results, I create a skewed-normal probability distribution and use it to assign probabilities to each bracket.

## Future Development

There are a few things I would like to implement in the future to make this tool more useful. The first one would be to make the code flexible enough to analyze all the tweet markets on PredictIt, there are currently four of them. The second would be to automate the scraping of tweets. Finally I plan on working more on the predictions and prices to create clear buying and selling signals.
