#!/usr/bin/env python
# coding: utf-8

# # *Research on Keyword using Twitter Data*
# 

# # 1.tweepy

# # Seems tweepy package still not fix with python 3.7
# # Here I'm using python 3.6.6 and upgrage the tweepy to the leatest version by 
# # using the following command on Anaconda Prompt:
# # pip3 install --upgrade git+https://github.com/tweepy/tweepy.git

# In[2]:


#from tweepy import Stream
#from tweepy import OAuthHandler
#from tweepy.streaming import StreamListener
#from __future__ import absolute_import, print_function

#comsumer_key = 'WVtv2RK0lKcEpc4bjW7ZX276r'
#comsumer_secret = 'PYdmwbf55hthHHcBGUNKTwoDDLeXlJITGaOV3shtCgOIjNZexb'
#access_token = '2762799427-LfgXdY7CWUVlfVbkh9idPdPKOc2rgbVVNwOxc4k'
#access_token_secret = '6UC2FHDR2MfLZiz6AFNWlWf597us79nOYmtLlJhj06shH'

#class StdOutListener(StreamListener):
#    def on_data(self,data):
#        try:
#            print(self,data)
#            with open('TwitterAPI.csv', 'a') as f:
#                f.write(data)
#        except:
#            pass
#        
#    def on_error(self,status):
#        print(status)
        
#l = StdOutListener()
#auth = OAuthHandler(comsumer_key, comsumer_secret)
#auth.set_access_token(access_token, access_token_secret)
#stream = Stream(auth, l)
#stream.filter(track=['iphone xs', 'ipad pro', 'macbook air'])


# In[7]:


import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import datetime
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
import numpy as np
import collections
from textblob import TextBlob


# # 2. Data Mining

# In[ ]:


tweets_data_path = 'TwitterAPI.csv'
#tweets_data_path = 'TwitterTest.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
print (len(tweets_data))


# # Tweets by language

# In[2]:


tweets = pd.DataFrame()
tweets['text'] = [x.get('text', None) for x in tweets_data]
tweets['lang'] = [x.get('lang', None) for x in tweets_data]
tweets['country'] = [x.get('place').get('country', None) if x.get('place', None) != None else None for x in tweets_data]

tweets_by_lang = tweets['lang'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Languages', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')
#plt.savefig('top5Lang.png')
plt.show()


# # Tweets by country
# # current issus:
# # no idea what's wrong with the fourth country's name

# In[60]:


tweets_by_country = tweets['country'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Countries', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 countries', fontsize=15, fontweight='bold')
tweets_by_country[:5].plot(ax=ax, kind='bar', color='blue')
#plt.savefig('top5Country.png')
plt.show()


# # Rank

# In[102]:


def check(x, matchWord):
    if x is None:
        return False
    text = x.lower()
    if (matchWord in text):
        return True
    else:
        return False

tweets['iphone xs'] = tweets['text'].apply(lambda x: check(x, 'iphone xs'))
tweets['ipad pro'] = tweets['text'].apply(lambda x: check(x, 'ipad pro'))
tweets['macbook air'] = tweets['text'].apply(lambda x: check(x, 'macbook air'))
print(tweets['iphone xs'].value_counts()[True])
print(tweets['ipad pro'].value_counts()[True])
print(tweets['macbook air'].value_counts()[True])

prg_langs = ['Iphone Xs', 'Ipad Pro', 'Macbook Air']
tweets_by_prg_lang = [tweets['iphone xs'].value_counts()[False], tweets['ipad pro'].value_counts()[True], tweets['macbook air'].value_counts()[True]]

x_pos = list(range(len(prg_langs)))
width = 0.3
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_prg_lang, width, alpha=1, color='gold')

# Setting axis labels and ticks
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: flu vs. cough vs. fever (Raw data)', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(prg_langs)
plt.grid()
#plt.savefig('rank.png')
plt.show()


# # 3.Geology Time Analysis

# In[198]:


def dataClean(fileName):
    tweets_data = []
    tweets_file = open(fileName, "r")
    count = 0
    for line in tweets_file:
        try:
            temp = json.loads(line)
            if 'iPhone' in temp['text']:
                continue
            tweet = {}
            #tweet['geo'] = temp['geo']
            #tweet['id'] = temp['id']
            #tweet['place'] = temp['place']
            tweet['text'] = temp['text']
            #Tweet stream problem. Can't find the key 'location'
            #tweet['location'] = temp['location']
            tweet['timestamp_ms'] = temp['timestamp_ms']
            tweet['lang'] = temp['lang']
            tweets_data.append(tweet)
            count += 1
        except:
            continue
    print(len(tweets_data))
    print(count)
    return tweets_data

def dataCount(path):
    tweets_file = open(path, 'r')
    count = 0
    for line in tweets_file:
        try:
            t = json.loads(line)
            if "location" in t:
                count += 1
        except:
            continue
    print(count)
#Always return 0
    
def readData(file):
    tweets_data = []
    tweets_file = open(file, 'r')
    for line in tweets_file:
        try:
            t = json.loads(line)
            tweets_data.append(t)
        except:
            continue
    print(len(tweets_data))
    return tweets_data

def write_file(data, fileName):
    with open(fileName, 'w') as outfile:
        for val in data:
            json.dump(val, outfile)
            outfile.write('\n')


# # Current Issus:
# # Can't find the key 'location' in raw data, looking for solution.

# In[197]:


geoPath = 'Test.json'
#dataPath = 'TwitterAPI.csv'
iphonePath = 'iphone.json'

dataCount(geoPath)

#tweets_data = cleanData(geoPath)
#write_file(dataClean(geoPath), iphonePath)
#tweets_data = readData(iphonePath)


# # Geo Analysis

# # Source code from matplotlib's github
# # https://github.com/matplotlib/basemap/blob/decfa95124dab76499734145d03b002b9db27477/examples/fillstates.py

# In[204]:


def plotGeo(stateDict):
    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    # draw state boundaries.
    shp_info = m.readshapefile('./st99_d00','states',drawbounds=True)
    # choose a color for each state based on population density.
    colors={}
    statenames=[]
    cmap = plt.cm.Greens # use 'hot' colormap
    # cmap = plt.cm.coolwarm
    vmin = 0; vmax = 14000 # set range.

    for shapedict in m.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia','Puerto Rico']:
            pop = stateDict[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            # colors[statename] = cmap(np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
            colors[statename] = cmap(pop * 1.5/ vmax)[:3]
        statenames.append(statename)
    # cycle through state names, color each one.
    ax = plt.gca() # get current axes instance
    ATOLL_CUTOFF = 0.005
    for ind,shapedict in enumerate(m.states_info):
        seg = m.states[int(shapedict['SHAPENUM'] - 1)]
        # skip DC and Puerto Rico.
        if statenames[ind] not in ['Puerto Rico', 'District of Columbia']:
        # Offset Alaska and Hawaii to the lower-left corner. 
            if statenames[ind] == 'Alaska':
            # Alaska is too big. Scale it down to 35% first, then transate it. 
                seg = list([(0.30*x_y[0] + 1100000, 0.30*x_y[1]-1300000) for x_y in seg])
            if shapedict['NAME'] == 'Hawaii' and float(shapedict['AREA']) > ATOLL_CUTOFF:
                seg = list([(x_y1[0] + 5200000, x_y1[1]-1400000) for x_y1 in seg])

            color = rgb2hex(colors[statenames[ind]]) 
            poly = Polygon(seg,facecolor=color,edgecolor='black',linewidth=.5)
            ax.add_patch(poly)
    sm = plt.cm.ScalarMappable(cmap="Greens", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_ticks([0, 3000, 6000, 9000, 14000])
    cbar.set_ticklabels(['0', '1500', '4000', '6000', '10000 or more'])
    # cbar.set_ticklabels(['0', '200', '500', '1000', '3600'])
    plt.title('Filling State Polygons by Density')
    # plt.savefig('geoDistribution.png')
    plt.show()
    
def getGeoDict(tweets):
    geoTweets = []
    geoDict = {
    'New Jersey':  438.00,
    'Rhode Island':   387.35,
    'Massachusetts':   312.68,
    'Connecticut':    271.40,
    'Maryland':   209.23,
    'New York':    155.18,
    'Delaware':    154.87,
    'Florida':     114.43,
    'Ohio':  107.05,
    'Pennsylvania':  105.80,
    'Illinois':    86.27,
    'California':  83.85,
    'Hawaii':  72.83,
    'Virginia':    69.03,
    'Michigan':    67.55,
    'Indiana':    65.46,
    'North Carolina':  63.80,
    'Georgia':     54.59,
    'Tennessee':   53.29,
    'New Hampshire':   53.20,
    'South Carolina':  51.45,
    'Louisiana':   39.61,
    'Kentucky':   39.28,
    'Wisconsin':  38.13,
    'Washington':  34.20,
    'Alabama':     33.84,
    'Missouri':    31.36,
    'Texas':   30.75,
    'West Virginia':   29.00,
    'Vermont':     25.41,
    'Minnesota':  23.86,
    'Mississippi':   23.42,
    'Iowa':  20.22,
    'Arkansas':    19.82,
    'Oklahoma':    19.40,
    'Arizona':     17.43,
    'Colorado':    16.01,
    'Maine':  15.95,
    'Oregon':  13.76,
    'Kansas':  12.69,
    'Utah':  10.50,
    'Nebraska':    8.60,
    'Nevada':  7.03,
    'Idaho':   6.04,
    'New Mexico':  5.79,
    'South Dakota':  3.84,
    'North Dakota':  3.59,
    'Montana':     2.39,
    'Wyoming':      1.96,
    'Alaska':     0.42}
    for key in geoDict:
        geoDict[key] = 0
    for tweet in tweets:
        if tweet['location']['country'] == "United States":
            if 'state' in tweet['location']:
                if tweet['location']['state'] in geoDict:
                    geoDict[tweet['location']['state']] += 1
                    geoTweets.append(tweet)
    return geoDict, geoTweets

def geoExp(dataPath = 'cleanData.json'):
    tweets = readData(dataPath)
    geoDict, geoTweets = getGeoDict(tweets)
    vmin = 0
    vmax = 0
    for key in geoDict:
        print(key, geoDict[key])
        if geoDict[key] > vmax:
            vmax = geoDict[key]
    # print vmax
    print('geo tweets:', np.sum([geoDict[key] for key in geoDict]))
    plotGeoDict(geoDict)
    plotGeo(geoDict)

def plotGeoDict(dataDict):
    od = collections.OrderedDict(sorted(dataDict.items()))
    plt.bar(list(range(len(list(od.keys())))), list(od.values()), 0.8, color='g')
    plt.xticks(list(range(len(list(od.keys())))), list(od.keys()), rotation='vertical')
    plt.rc('xtick', labelsize=10)
    plt.title('geo distribution')
    # plt.savefig('geoDict.png')
    plt.show()
    
geoExp()


# # Time Analysis

# In[203]:


def getTimeDict(dataPath = 'cleanData.json'):
    tweets = readData(dataPath)
    geoDict, tweets = getGeoDict(tweets)
    dateDict = {}
    hourDict = {}
    for i in range(len(tweets)):
        date = datetime.datetime.fromtimestamp(int(tweets[i]['timestamp_ms']) / 1000).strftime('%m-%d')
        hour = datetime.datetime.fromtimestamp(int(tweets[i]['timestamp_ms']) / 1000).strftime('%m-%d-%H')
        if date in dateDict:
            dateDict[date] += 1
        else:
            dateDict[date] = 1
        if date not in hourDict:
            hourDict[date] = {}
            hourDict[date][hour] = 1
        else:
            if hour not in hourDict[date]:
                hourDict[date][hour] = 1
            else:
                hourDict[date][hour] += 1
    return dateDict, hourDict

def plotTimeDict(dataDict):
    od = collections.OrderedDict(sorted(dataDict.items()))
    plt.bar(list(range(len(list(od.keys())))), list(od.values()), 0.5, color='g')
    plt.xticks(list(range(len(list(od.keys())))), [val[-2:] for val in list(od.keys())])
    plt.title(list(od.items())[0][0][:-3])
    # plt.savefig('date_'+od.items()[0][0][:-3]+'.png')
    plt.show()

def timeExp(dataPath = 'cleanData.json'):
    dateDict, hourDict = getTimeDict(dataPath)
    plotTimeDict(dateDict)
    for key in hourDict:
        plotTimeDict(hourDict[key])
        
timeExp()


# # Sentiment Analysis

# In[218]:


def getSentDict(dataPath = 'cleanData.json'):
    tweets = readData(dataPath)
    geoDict, tweets = getGeoDict(tweets)
    sentDict = {'positive': 0, 'negative': 0,'neutral': 0 }

    for i in range(len(tweets)):
        testimonial = TextBlob(tweets[i]['text'])
        if testimonial.sentiment.polarity > 0.1:
            sentDict['positive'] += 1
        elif testimonial.sentiment.polarity < -0.1:
            sentDict['negative'] += 1
        else:
            sentDict['neutral'] += 1
    return sentDict

def plotSentDict(sentDict):
    f, axarr = plt.subplots(1,2)
    axarr[0].bar(list(range(len(list(sentDict.keys())))), list(sentDict.values()), 0.5, color='blue')
    axarr[0].set_xticks(list(range(len(list(sentDict.keys())))))
    axarr[0].set_xticklabels(list(sentDict.keys()))
    axarr[0].set_title('polarity')
    axarr[1].pie([sentDict[key] for key in sentDict], labels=[key for key in sentDict], autopct='%1.1f%%')
    axarr[1].axis('equal')
    plt.suptitle('sentiment analysis')
    # plt.savefig('sentiment.png')
    plt.show()
    
def sentExp():
    sentDict = getSentDict()
    plotSentDict(sentDict)

    
sentExp()

