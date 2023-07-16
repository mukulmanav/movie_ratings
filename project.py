#%%
#In this project we proved that rating on the movie sites is more than the actual ratings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df=pd.read_csv("fandango_scrape.csv")
# %%
df.head()
# %%
df.info()
# %%
df.describe()
# %%
#scatterplot showing relationship between popularity of a film and its rating
plt.figure(figsize=(10,4),dpi=150)
sns.scatterplot(data=df,x="RATING",y="VOTES")
# %% Creating a new column-YEAR
df['YEAR']=df['FILM'].apply(lambda title:title.split('(')[-1].replace(')',''))

# %%
df
# %% movies are in the Fandango DataFrame per year
df['YEAR'].value_counts()
# %%
sns.countplot(data=df,x='YEAR')
# %%  top 10 movies with highest number of votes
df.nlargest(10,'VOTES')

# %% how many movies are with Zero votes
len(df[df['VOTES']==0])

# %%
fan_review=df[df['VOTES']>0]
# %%
fan_review
# %%reate a KDE plot (or multiple kdeplots) that displays the distribution of ratings that are displayed (STARS) versus what the true rating was from votes (RATING). Clip the KDEs to 0-5.
sns.kdeplot(data=fan_review,x="RATING",clip=[0,5],fill=True,label='True Rating')
sns.kdeplot(data=fan_review,x="STARS",clip=[0,5],fill=True,label='Stars Displayed')

plt.legend(loc=(1.05,0.5))
# %% Created a new column of the different between STARS and RATING
fan_review["DIFFERENCE"]=fan_review['STARS']-fan_review['RATING']
# %%
fan_review
# %%
fan_review['DIFFERENCE']=fan_review['DIFFERENCE'].round(2)
# %%
fan_review
# %% Create a count plot to display the number of times a certain difference occurs
plt.figure(figsize=(12,4),dpi=150)
sns.countplot(data=fan_review,x='DIFFERENCE',palette='magma')
# %%
all_sites=pd.read_csv("all_sites_scores.csv")
# %%
all_sites.head()
# %%
all_sites.describe()
# %% Create a scatterplot exploring the relationship between RT Critic reviews and RT User reviews
sns.scatterplot(data=all_sites,x="RottenTomatoes",y='RottenTomatoes_User')
# %%
all_sites['ROTTEN_DIFF']=all_sites['RottenTomatoes']-all_sites['RottenTomatoes_User']
# %%
all_sites
# %% Calculate the Mean Absolute Difference between RT scores and RT User scores 
all_sites['ROTTEN_DIFF'].apply(abs).mean()
# %% Plot the distribution of the differences between RT Critics Score and RT User Score. There should be negative values in this distribution plot. Feel free to use KDE or Histograms to display this distribution.
plt.figure(figsize=(10,4),dpi=200)
sns.histplot(data=all_sites,x='ROTTEN_DIFF',kde=True,bins=25)
plt.title("RT Critics Score minus RT User Score")
# %% What are the top 5 movies users rated higher than critics on average
print("Users Love but Critics Hate")
all_sites.nsmallest(5,'ROTTEN_DIFF')[['FILM','ROTTEN_DIFF']]
# %% show the top 5 movies critics scores higher than users on average.
print("Critics love, but Users Hate")
all_sites.nlargest(5,'ROTTEN_DIFF')[['FILM','ROTTEN_DIFF']]
# %% Display a scatterplot of the Metacritic Rating versus the Metacritic User rating

plt.figure(figsize=(10,4),dpi=150)
sns.scatterplot(data=all_sites,x='Metacritic',y='Metacritic_User')
plt.xlim(0,100)
plt.ylim(0,10)
# %% Create a scatterplot for the relationship between vote counts on MetaCritic versus vote counts on IMDB
plt.figure(figsize=(10,4),dpi=150)
sns.scatterplot(data=all_sites,x='Metacritic_user_vote_count',y='IMDB_user_vote_count')
# %% What movie has the highest IMDB user vote count?
all_sites.nlargest(1,'IMDB_user_vote_count')
# %%  What movie has the highest Metacritic User Vote count?
all_sites.nlargest(1,'Metacritic_user_vote_count')
# %% Not every movie in the Fandango table is in the All Sites table, since some Fandango movies have very little or no reviews. We only want to compare movies that are in both DataFrames, so do an *inner* merge to merge together both DataFrames based on the FILM columns.
dff = pd.merge(df,all_sites,on='FILM',how='inner')
# %%
dff.head()
# %% Notice that RT,Metacritic, and IMDB don't use a score between 0-5 stars like Fandango does. In order to do a fair comparison, we need to *normalize* these values so they all fall between 0-5 stars and the relationship between reviews stays the same
# Keep in mind, a simple way to convert ratings:
# * 100/20 = 5 
# * 10/2 = 5

# Dont run this cell multiple times, otherwise you keep dividing!
dff['RT_Norm'] = np.round(dff['RottenTomatoes']/20,1)
dff['RTU_Norm'] =  np.round(dff['RottenTomatoes_User']/20,1)

# Dont run this cell multiple times, otherwise you keep dividing!
dff['Meta_Norm'] =  np.round(dff['Metacritic']/20,1)
dff['Meta_U_Norm'] =  np.round(dff['Metacritic_User']/2,1)

# Dont run this cell multiple times, otherwise you keep dividing!
dff['IMDB_Norm'] = np.round(dff['IMDB']/2,1)
# %%
dff.head()

# %% create a norm_scores DataFrame that only contains the normalizes ratings. Include both STARS and RATING from the original Fandango table.
norm_scores = dff[['STARS','RATING','RT_Norm','RTU_Norm','Meta_Norm','Meta_U_Norm','IMDB_Norm']]
# %%
def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)
# %%
fig, ax = plt.subplots(figsize=(15,6),dpi=150)
sns.kdeplot(data=norm_scores,clip=[0,5],shade=True,palette='Set1',ax=ax)
move_legend(ax, "upper left")
# %%
fig, ax = plt.subplots(figsize=(15,6),dpi=150)
sns.kdeplot(data=norm_scores[['RT_Norm','STARS']],clip=[0,5],shade=True,palette='Set1',ax=ax)
move_legend(ax, "upper left")
# %% Create a histplot comparing all normalized scores
plt.subplots(figsize=(15,6),dpi=150)
sns.histplot(norm_scores,bins=50)

# %%
norm_films = dff[['STARS','RATING','RT_Norm','RTU_Norm','Meta_Norm','Meta_U_Norm','IMDB_Norm','FILM']]
# %%
norm_films.nsmallest(10,'RT_Norm')
# %%
print('\n\n')
plt.figure(figsize=(15,6),dpi=150)
worst_films = norm_films.nsmallest(10,'RT_Norm').drop('FILM',axis=1)
sns.kdeplot(data=worst_films,clip=[0,5],shade=True,palette='Set1')
plt.title("Ratings for RT Critic's 10 Worst Reviewed Films");
# %%
norm_films.iloc[25]
# %%
