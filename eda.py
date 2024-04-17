import numpy as np
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)

kInputDirectory = (r'C:\Users\khilr\Desktop\IS5126FinalProject\ag_news_clean')
kOutputDirectory = (r"C:\Users\khilr\Desktop\IS5126FinalProject\Figures")

train_df = pd.read_csv(kInputDirectory+r'\train.csv')
test_df = pd.read_csv(kInputDirectory+r'\test.csv')

train_df['polarity'] = train_df['cleaned_text'].map(lambda text: TextBlob(text).sentiment.polarity)
train_df['length'] = train_df['cleaned_text'].astype(str).apply(len)
train_df['wordCount'] = train_df['cleaned_text'].apply(lambda x:len(str(x).split()))

print(train_df.head(10))
print(train_df.describe())
print(train_df.info())

def getTopNWords(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bagOfWords = vec.transform(corpus)
    sumWords = bagOfWords.sum(axis = 0)
    wordFrequency = [(word, sumWords[0, idx]) for word, idx in vec.vocabulary_.items()]
    wordFrequency = sorted(wordFrequency, key= lambda x: x[1], reverse = True)
    return wordFrequency[:n]

commonBeforeN = getTopNWords(train_df['text'], 20)
commonAfterN = getTopNWords(train_df['cleaned_text'], 20)

commonFor0 = getTopNWords(train_df.loc[train_df['label']==0]['cleaned_text'], 20)
commonFor1 = getTopNWords(train_df.loc[train_df['label']==1]['cleaned_text'], 20)
commonFor2 = getTopNWords(train_df.loc[train_df['label']==2]['cleaned_text'], 20)
commonFor3 = getTopNWords(train_df.loc[train_df['label']==3]['cleaned_text'], 20)

common0 = pd.DataFrame(commonFor0, columns=['text', 'count'])
common1 = pd.DataFrame(commonFor1, columns=['text', 'count'])
common2 = pd.DataFrame(commonFor2, columns=['text', 'count'])
common3 = pd.DataFrame(commonFor3, columns=['text', 'count'])

commonFinal0 = common0.groupby('text', as_index=False).sum().sort_values('count', ascending=False)
commonFinal1 = common1.groupby('text', as_index=False).sum().sort_values('count', ascending=False)
commonFinal2 = common2.groupby('text', as_index=False).sum().sort_values('count', ascending=False)
commonFinal3 = common3.groupby('text', as_index=False).sum().sort_values('count', ascending=False)

trainBefStop = pd.DataFrame(commonBeforeN, columns= ['text', 'count'])
trainBeforeStop = trainBefStop.groupby('text', as_index=False).sum().sort_values('count', ascending=False)

trainAftStop = pd.DataFrame(commonAfterN, columns= ['text', 'count'])
trainAfterStop = trainAftStop.groupby('text', as_index=False).sum().sort_values('count', ascending=False)

def getTopBigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2,2), stop_words='english').fit(corpus)
    bagOfWords = vec.transform(corpus)
    sumWords = bagOfWords.sum(axis = 0)
    wordFrequency = [(word, sumWords[0, idx]) for word, idx in vec.vocabulary_.items()]
    wordFrequency = sorted(wordFrequency, key= lambda x: x[1], reverse = True)
    return wordFrequency[:n]

commonBeforeBigram = getTopBigram(train_df['text'], 20)
commonAfterBigram = getTopBigram(train_df['cleaned_text'], 20)

trainBefBigram = pd.DataFrame(commonBeforeBigram, columns=['text', 'count'])
trainBeforeBigram = trainBefBigram.groupby('text', as_index=False).sum().sort_values('count', ascending=False)
trainAftBigram = pd.DataFrame(commonAfterBigram, columns=['text', 'count'])
trainAfterBigram = trainAftBigram.groupby('text', as_index=False).sum().sort_values('count', ascending=False)

def getTopTrigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3,3), stop_words='english').fit(corpus)
    bagOfWords = vec.transform(corpus)
    sumWords = bagOfWords.sum(axis = 0)
    wordFrequency = [(word, sumWords[0, idx]) for word, idx in vec.vocabulary_.items()]
    wordFrequency = sorted(wordFrequency, key= lambda x: x[1], reverse = True)
    return wordFrequency[:n]

commonBeforeTrigram = getTopTrigram(train_df['text'], 20)
commonAfterTrigram = getTopTrigram(train_df['cleaned_text'], 20)

trainBefTrigram = pd.DataFrame(commonBeforeTrigram, columns=['text', 'count'])
trainBeforeTrigram = trainBefTrigram.groupby('text', as_index=False).sum().sort_values('count', ascending=False)
trainAftTrigram = pd.DataFrame(commonAfterTrigram, columns=['text', 'count'])
trainAfterTrigram = trainAftTrigram.groupby('text', as_index=False).sum().sort_values('count', ascending=False)

def plotHistogram(data, bins, colormap, xlabel, ylabel, title, filename):
    plt.figure()  # Start a new figure
    n, bins, patches = plt.hist(data, bins=bins, edgecolor='white')

    # Color mapping
    cm = plt.cm.get_cmap(colormap)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col) 

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c)) 

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(kOutputDirectory + filename)
    plt.close()

plotHistogram(train_df['wordCount'], 50, 'magma', 'Word Count', 'Count', 'Word Count Distribution', '\\word_count.png')
plotHistogram(train_df['label'], 50, 'inferno', 'Labels', 'Count', 'Label Distribution', '\\labels.png')
plotHistogram(train_df['polarity'], 50, 'viridis', 'Polarity', 'Count', 'Sentiment Polarity Distribution', '\\polarity_histogram.png')
plotHistogram(train_df['length'], 50, 'plasma', 'Character Length', 'Count', 'News Characters Distribution', '\\news_length.png')

def plotBarChart(data, colormap, xlabel, ylabel, title, filename):
    plt.figure()
    colors = plt.cm.get_cmap(colormap, len(data))
    bars = plt.bar(data['text'], data['count'], color=colors(range(len(data))), edgecolor='white')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.xticks(rotation=90)
    
    plt.tight_layout()  
    plt.savefig(kOutputDirectory + filename)
    plt.close()


plotBarChart(trainBeforeStop, 'viridis', 'Text', 'Count', 'Top 20 Words Before Removing Stopwords', '\\top20BeforeStopwords.png')
plotBarChart(trainAfterStop, 'viridis', 'Text', 'Count', 'Top 20 Words After Removing Stopwords', '\\top20AfterStopwords.png')
plotBarChart(trainBeforeBigram, 'viridis', 'Text', 'Count', 'Top 20 Bigrams Before Removing Stopwords', '\\top20BeforeBigrams.png')
plotBarChart(trainAfterBigram, 'viridis', 'Text', 'Count', 'Top 20 Bigrams After Removing Stopwords', '\\top20AfterBigrams.png')
plotBarChart(trainBeforeTrigram, 'viridis', 'Text', 'Count', 'Top 20 Trigrams Before Removing Stopwords', '\\top20BeforeTrigrams.png')
plotBarChart(trainAfterTrigram, 'viridis', 'Text', 'Count', 'Top 20 Trigrams After Removing Stopwords', '\\top20AfterTrigrams.png')

plotBarChart(commonFinal0, 'magma', 'Word', 'Count', 'Top 20 Words for World News', '\\wordsWorld.png')
plotBarChart(commonFinal1, 'magma', 'Word', 'Count', 'Top 20 Words for Sports News', '\\wordsSports.png')
plotBarChart(commonFinal2, 'magma', 'Word', 'Count', 'Top 20 Words for Business News', '\\wordsBusiness.png')
plotBarChart(commonFinal3, 'magma', 'Word', 'Count', 'Top 20 Words for Sci/Tech News', '\\wordsSciTech.png')