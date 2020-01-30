# import wheels
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

df_yout = pd.read_csv("../input/youtube-new/USvideos.csv")

#%% drop_duplicate
df_yout = df_yout.drop_duplicates(subset=['video_id','trending_date'], keep='last', inplace=False) #duble drop

df_yout['trending_times'] = np.nan #compute trending_times
for v_id in df_yout['video_id'].unique():
    trending_times = sum(df_yout['video_id'] == v_id)
    df_yout.loc[(df_yout["video_id"] == v_id),"trending_times"] = trending_times

df_yout = df_yout.drop_duplicates(subset='video_id', keep='last', inplace=False) #drop

#%% Category_ID to Category_name

df_yout['category_name'] = np.nan

df_yout.loc[(df_yout["category_id"] == 1), "category_name"] = 'Film and Animation'
df_yout.loc[(df_yout["category_id"] == 2), "category_name"] = 'Cars and Vehicles'
df_yout.loc[(df_yout["category_id"] == 10), "category_name"] = 'Music'
df_yout.loc[(df_yout["category_id"] == 15), "category_name"] = 'Pets and Animals'
df_yout.loc[(df_yout["category_id"] == 17), "category_name"] = 'Sport'
df_yout.loc[(df_yout["category_id"] == 19), "category_name"] = 'Travel and Events'
df_yout.loc[(df_yout["category_id"] == 20), "category_name"] = 'Gaming'
df_yout.loc[(df_yout["category_id"] == 22), "category_name"] = 'People and Blogs'
df_yout.loc[(df_yout["category_id"] == 23), "category_name"] = 'Comedy'
df_yout.loc[(df_yout["category_id"] == 24), "category_name"] = 'Entertainment'
df_yout.loc[(df_yout["category_id"] == 25), "category_name"] = 'News and Politics'
df_yout.loc[(df_yout["category_id"] == 26), "category_name"] = 'How to and Style'
df_yout.loc[(df_yout["category_id"] == 27), "category_name"] = 'Education'
df_yout.loc[(df_yout["category_id"] == 28), "category_name"] = 'Science and Technology'
df_yout.loc[(df_yout["category_id"] == 29), "category_name"] = 'Non Profits and Activism'
df_yout.loc[(df_yout["category_id"] == 25), "category_name"] = 'News & Politics'

#%% log

df_yout['likes_log'] = np.log(df_yout['likes'] + 1)
df_yout['views_log'] = np.log(df_yout['views'] + 1)
df_yout['dislikes_log'] = np.log(df_yout['dislikes'] + 1)
df_yout['comment_log'] = np.log(df_yout['comment_count'] + 1)

#%% Date and time (weekday and hour)

df_yout['publish_date']=df_yout['publish_time'].apply(lambda x : x.split('T')[0])
df_yout['weekday']=pd.to_datetime(df_yout['publish_date']).dt.weekday

df_yout['publish_time']=df_yout['publish_time'].apply(lambda x : x.split('T')[1])
df_yout['publish_hour']=df_yout['publish_time'].apply(lambda x : x[0:2])