import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import numpy as np
import urllib2
import json, re
from scipy import stats
from scipy.stats import skewnorm

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='bhines239', api_key='fKAY4BXT5CHgakY2KB1Z')


###############################
## Data Manipulation section ##
###############################

# Tweet collection

startDay = pd.to_datetime("2017-01-25 16:00:00", utc=1).tz_convert('US/Eastern')
now = pd.to_datetime(pd.datetime.now() + pd.Timedelta('4h'), utc = 1).tz_convert('US/Eastern')
endDay = startDay + pd.Timedelta(np.floor(((now - startDay) / np.timedelta64(1, 'D'))/7)*7, unit = 'd')

tweets = pd.read_csv('trumpTwitterArchive.csv')

tweets['created_at'] = pd.to_datetime(tweets['created_at'])
tweets['created_at'] = tweets['created_at'].dt.tz_localize('GMT').dt.tz_convert('US/Eastern')
tweets['fromStart'] = tweets['created_at'] - startDay
tweets = tweets.loc[tweets['created_at'] > startDay]
tweets['daysFromStart'] = (tweets['fromStart'] / np.timedelta64(1, 'D')).apply(np.floor)
tweets['hoursFromStart'] = (tweets['fromStart'] / np.timedelta64(1, 'h')).apply(np.floor)
tweets['weekNumber'] = ((tweets['daysFromStart']/7) + 1).apply(np.floor)
tweets['dayNumber'] = tweets['daysFromStart'] + 1 - (tweets['weekNumber'] -1)*7
tweets['hourNumber'] = tweets['hoursFromStart'] - (tweets['weekNumber'] -1)*168

thisWeek = tweets.loc[tweets['created_at'] > endDay]
tweets = tweets.loc[tweets['created_at'] < endDay]

#Price collection
url = "https://www.predictit.org/api/marketdata/ticker/RDTS." + str(endDay.month).rjust(2, '0') + str(endDay.day + 7).rjust(2, '0') + str(endDay.year)[-2:]

page = re=json.load(urllib2.urlopen(url))
page = pd.DataFrame.from_dict(page)

priceData = []
for i in range(0, len(page)):
	data = pd.DataFrame.from_dict(page.Contracts[i],  orient='index').transpose()
	priceData.append(data)

priceData = pd.concat(priceData, axis=0, ignore_index=True)

priceData = priceData.loc[:,('Name', 'BestBuyYesCost', 'BestSellYesCost', 'BestBuyNoCost', 'BestSellNoCost','TickerSymbol')]
priceData.sort_values(by = 'TickerSymbol', ascending=True, inplace=True)
priceData.reset_index(drop=True, inplace=True)

brackets = []
for b in range(1,len(priceData)):
	brackets.append(int(priceData['TickerSymbol'][b].split(".")[0]))

#Rate collection
d = []
for h in range(0,168, 1):
	tmptweets = tweets.loc[tweets['hourNumber'] >= h].groupby('weekNumber').size()
	
	if(len(tmptweets) < tweets['weekNumber'].max()):
		tmptweets = np.append(tmptweets.values, (int(tweets['weekNumber'].max() - len(tmptweets)) * [0]))
		
	d.append({'hourNumber': h,
			  'meanRemainingTweets': tmptweets.mean(),
			  'stdRemainingTweets': tmptweets.std(),
			  'skewRemainingTweets': stats.skew(tmptweets)})
	
hourlyRates = pd.DataFrame(d)

d = []
for h in range(0,168, 1):
	tmptweets = tweets.loc[tweets['hourNumber'] <= h].groupby('weekNumber').size()

	if(len(tmptweets) < tweets['weekNumber'].max()):
		tmptweets = np.append(tmptweets.values, (int(tweets['weekNumber'].max() - len(tmptweets)) * [0]))
		
	d.append({'hourNumber': h,
			  'meanTweets': tmptweets.mean(),
			  'stdTweets': tmptweets.std()})

averagePace = pd.DataFrame(d)

#Prection updates
thisWeek = thisWeek[['text', 'created_at','fromStart', 'daysFromStart', 'weekNumber',
					 'dayNumber', 'hourNumber']]
dummy = pd.DataFrame({'text': 'Dummy Tweet',
					  'created_at': pd.datetime.now(), 
					  'fromStart': now - startDay,
					  'daysFromStart': np.floor((now - startDay)/np.timedelta64(1, 'D')),
					  'weekNumber': thisWeek['weekNumber'][0],
					  'dayNumber': np.floor((now - startDay)/np.timedelta64(1, 'D'))+ 1 - (thisWeek['weekNumber'][0] -1)*7,
					  'hourNumber': np.floor((now - startDay)/np.timedelta64(1, 'h'))+ 1 - (thisWeek['weekNumber'][0] -1)*168},
					 index=[0])

thisWeek = pd.concat([dummy,thisWeek.iloc[:]]).reset_index(drop=True)

lastPrediction = thisWeek['text'].count() - 1 + hourlyRates.loc[hourlyRates['hourNumber'] == thisWeek['hourNumber'][0],'meanRemainingTweets'].iloc[0]
stddev = hourlyRates.loc[hourlyRates['hourNumber'] == thisWeek['hourNumber'][0],'stdRemainingTweets'].iloc[0]
skew = hourlyRates.loc[hourlyRates['hourNumber'] == thisWeek['hourNumber'][0],'skewRemainingTweets'].iloc[0]


yesProbs = []
for b in range(0,len(priceData)):
    if b == 0:
        yesProbs.append(skewnorm.cdf(brackets[b],skew,lastPrediction,stddev))
    elif b > 0 and b < priceData.index.max():
        yesProbs.append(1 - skewnorm.cdf(brackets[b-1],skew,lastPrediction,stddev) - skewnorm.sf(brackets[b],skew,lastPrediction,stddev))
    else:
        yesProbs.append(skewnorm.sf(brackets[b-1],skew,lastPrediction,stddev))

priceData['probabilityYes'] = yesProbs
priceData['probabilityNo'] = 1 - priceData['probabilityYes']

#Plot data collection
plotdf = averagePace
plotdf['averagePlus'] = plotdf['meanTweets'] + plotdf['stdTweets']
plotdf['averageMinus'] = plotdf['meanTweets'] - plotdf['stdTweets']

thisWeekHourly = pd.DataFrame(thisWeek.hourNumber.value_counts().reset_index())
thisWeekHourly.columns = ['hourNumber', 'count']
thisWeekHourly.sort_values(by = 'hourNumber', inplace=True)
thisWeekHourly['currentCount'] = thisWeekHourly['count'].cumsum()
thisWeekHourly.loc[thisWeekHourly.index[-1], 'currentCount'] = thisWeekHourly.loc[thisWeekHourly.index[-1], 'currentCount'] - 1

plotdf = plotdf.merge(thisWeekHourly[['hourNumber', 'currentCount']], left_on='hourNumber', right_on='hourNumber', how='left')

plotdf.loc[:thisWeekHourly['hourNumber'].max(), :] = plotdf.loc[:thisWeekHourly['hourNumber'].max(), :].fillna(method='ffill')
plotdf.loc[:thisWeekHourly['hourNumber'].max(), :] = plotdf.loc[:thisWeekHourly['hourNumber'].max(), :].fillna(0)

plotdf['forecast'] = plotdf['currentCount']
plotdf['forecastPlus'] = plotdf['currentCount']
plotdf['forecastMinus'] = plotdf['currentCount']

plotdf.loc[plotdf.index[-1], 'forecast'] = hourlyRates.loc[hourlyRates['hourNumber'] == thisWeekHourly['hourNumber'].max(),'meanRemainingTweets'].iloc[0] + plotdf['currentCount'].max()
plotdf.loc[plotdf.index[-1], 'forecastPlus'] = plotdf.loc[plotdf.index[-1], 'forecast'] + hourlyRates.loc[hourlyRates['hourNumber'] == thisWeekHourly['hourNumber'].max(),'stdRemainingTweets'].iloc[0]
plotdf.loc[plotdf.index[-1], 'forecastMinus'] = plotdf.loc[plotdf.index[-1], 'forecast'] - hourlyRates.loc[hourlyRates['hourNumber'] == thisWeekHourly['hourNumber'].max(),'stdRemainingTweets'].iloc[0]

plotdf['forecast'] = plotdf['forecast'].interpolate()
plotdf['forecastPlus'] = plotdf['forecastPlus'].interpolate()
plotdf['forecastMinus'] = plotdf['forecastMinus'].interpolate()

#Plotting objects

upper_bound = go.Scatter(
	name='Upper Bound',
	x=plotdf['hourNumber'],
	y=plotdf['averagePlus'],
	mode='lines',
	marker=dict(color="444"),
	line=dict(width=0),
	fillcolor='rgba(68, 68, 68, 0.3)',
	fill='tonexty')

weeklyMean = go.Scatter(
	name='Average',
	x=plotdf['hourNumber'],
	y=plotdf['meanTweets'],
	mode='lines',
	line = dict(
		color = ('rgb(31, 119, 180)'),
		width = 2,
		dash = 'dash'),
	fillcolor='rgba(68, 68, 68, 0.3)',
	fill='tonexty')

lower_bound = go.Scatter(
	name='Lower Bound',
	x=plotdf['hourNumber'],
	y=plotdf['averageMinus'],
	marker=dict(color="444"),
	line=dict(width=0),
	mode='lines')

current = go.Scatter(
	name='Current Count',
	x = plotdf['hourNumber'],
	y = plotdf['currentCount'],
	mode='lines',
	line = dict(
		color = ('rgb(0, 0, 0)'),
		width = 2))
		
upper_forecast = go.Scatter(
	name='Upper Bound',
	x=plotdf['hourNumber'],
	y=plotdf['forecastPlus'],
	mode='lines',
	marker=dict(color="444"),
	line=dict(width=0),
	fillcolor='rgba(68, 68, 68, 0.3)',
	fill='tonexty')

meanForecast = go.Scatter(
	name='Forecast',
	x=plotdf['hourNumber'],
	y=plotdf['forecast'],
	mode='lines',
	line = dict(
		color = ('rgb(31, 119, 180)'),
		width = 2,
		dash = 'dash'),
	fillcolor='rgba(68, 68, 68, 0.3)',
	fill='tonexty')

lower_forecast = go.Scatter(
	name='Lower Bound',
	x=plotdf['hourNumber'],
	y=plotdf['forecastMinus'],
	marker=dict(color="444"),
	line=dict(width=0),
	mode='lines')

# Table generator

def generate_table(priceData, max_rows=9):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in priceData.columns])] +

        # Body
        [html.Tr([
            html.Td(priceData.iloc[i][col]) for col in priceData.columns
        ]) for i in range(min(len(priceData), max_rows))]
    )


#################
## APP section ##
#################

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Donnie Tweets'),
    html.Div(children='''
        Current prices and probabilites for the PredictIt @realDonald Trump Tweet Markets
    '''),
	html.Div([
	dcc.Graph(id='average-tweet-graph',
	figure = {'data': [lower_bound, weeklyMean, upper_bound, current],
		'layout':
		go.Layout(yaxis=dict(title='Number of Tweets'),
			xaxis=dict(title='Hour Number'),
			title='Current Tweet Count over Average Tweet Pace',
			showlegend = False)}),
    dcc.Graph(id='forecast-tweet-graph',
    figure = {'data': [lower_forecast, meanForecast, upper_forecast, current],
		'layout':
		go.Layout(yaxis=dict(title='Number of Tweets'),
            xaxis=dict(title='Hour Number'),
            title='Current Tweet Count With Average Remaining Pace',
            showlegend = False)})],style={'max-width': '900px'})
	,
	generate_table(priceData)
], style={"padding-bottom": "80px"})


app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


if __name__ == '__main__':
    app.run_server(debug=True)