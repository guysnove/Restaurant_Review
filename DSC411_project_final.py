# DSC 411 Final Project
# Chase and Guysnove
# there are multiple areas of commented code that can be uncommented to run.
# they are commented to speed up the processing


# It worked on jupyter notebooks, but this is where the comprehensive code is
# Therefore, to view that then you will need to uncomment the import and the code for it
# lines 42, 326 and 339

import nltk
import ssl
import re
import pandas as pd
import numpy as np
from gensim import utils
import gensim.parsing.preprocessing as gsp
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, TreebankWordDetokenizer
from sklearn import metrics
from textblob import TextBlob
from autocorrect import Speller
from statistics import mean
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
import seaborn as sns
from pyod.models.knn import KNN
#from wordcloud import WordCloud




# this was found to allow the lemmatizer to be downloaded and run
'''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('omw-1.4')
'''


# functions to make code cleaner
#------------------------------------------------------------------------
# read data and drop unneeded columns
def read_data(path):
    data = pd.read_csv(path, sep=',')
    selected_data = data.drop(columns=["No. 2", "No. 3", "Time"])
    text_data = data[["Review text"]]
    data["AVG Valence"] = (np.floor((data["Rater 1 - Valence"] + data["Rater 2 - Valence"]) / 2))
    data["AVG Arousal"] = (np.floor((data["Rater 1 - Arousal"] + data["Rater 2 - Arousal"]) / 2))
    target_r1v = pd.Categorical(data["Rater 1 - Valence"])
    target_r1a = pd.Categorical(data["Rater 1 - Arousal"])
    target_r2v = pd.Categorical(data["Rater 2 - Valence"])
    target_r2a = pd.Categorical(data["Rater 2 - Arousal"])
    target_avg_val = pd.Categorical(data["AVG Valence"])
    target_avg_ar = pd.Categorical(data["AVG Arousal"])

    return text_data, target_r1v, target_r1a, target_r2v, target_r2a, target_avg_val, target_avg_ar

# tokenize the text in the pandas dataframe
# also make everything lowercase

def tokenize(data):
    data["Review text"] = data["Review text"].apply(lambda row: [word_tokenize(row)])
    data["Review text"] = data["Review text"].apply(lambda row: [word for word in row for word in word])

    return data

# remove stopwords
def remove_stopwords(data):
    stop_words = stopwords.words("english")
    data["Review text"] = data["Review text"].apply(lambda row: ' '.join([word for word in row.split() if word not in (stop_words)]))

    return data

# remove numbers/punctuation
def remove_spec_char(data):
    data.replace('\d+', '', regex=True, inplace=True)
    data = data.apply(lambda row: [re.sub(r'[^\w\s]', '', word.lower()) for word in row])
    return data

# stem the words first
def stem(data):
    ps = PorterStemmer()
    data["Review text"] = data["Review text"].apply(lambda row: [ps.stem(word) for word in row])

    return data

# lemmatize the text
def lemmatize(data):
    lem = WordNetLemmatizer()
    data["Review text"] = data["Review text"].apply(lambda row: [lem.lemmatize(word, "v") for word in row])

    return data

# merge words back to invididual reviews
def merge_tokens(data):
    detoken = TreebankWordDetokenizer()
    data["Review text"] = data["Review text"].apply(lambda row: detoken.detokenize(row))

    return data

# correct spelling errors
def correct_spelling(data):
    check = Speller(lang='en')
    data['Review text'] = data["Review text"].apply(lambda row: (check(row)))

    return data

def vectorize(data):
    vec = CountVectorizer()
    data.rename(columns={"Cleaned text": "Cleaned_text"}, inplace=True)
    X = vec.fit_transform(data.Cleaned_text)
    frame = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

    return frame

def tfidf_vectorize(data):
    vec = TfidfVectorizer()
    data.rename(columns={"Cleaned text": "Cleaned_text"}, inplace=True)
    X = vec.fit_transform(data.Cleaned_text)
    terms = vec.get_feature_names_out()

    return X, terms


filters = [
    gsp.strip_tags,
    gsp.strip_punctuation,
    gsp.strip_multiple_whitespaces,
    gsp.strip_numeric,
    gsp.remove_stopwords,
    gsp.strip_short,
    gsp.stem_text
        ]

def clean_data(review):
    review = review.lower()
    review = utils.to_unicode(review)
    for f in filters:
        review = f(review)
    return review


def format_data(path):

    # read in data
    text_data, target_r1v, target_r1a, target_r2v, target_r2a, \
        target_avg_val, target_avg_ar = read_data(path)
    '''
    # remove numbers/ remove punctuation
    no_spec_char = remove_spec_char(text_data)
    #print(no_spec_char.head(10))

    # tokenize text
    tokenized_text = tokenize(no_spec_char)
    #print(tokenized_text.head(10))

    # lemmatize the text
    lemmatized_text = lemmatize(tokenized_text)
    #print(lemmatized_text.head(10))

    # merge tokens back together
    merged_text = merge_tokens(lemmatized_text)
    #print(merged_text.head(10))

    # remove stopwords
    text_no_sw = remove_stopwords(merged_text)
    #print(text_no_sw.head(10))

    # correct spelling this takes a long time to run
    #cleaned = correct_spelling(text_no_sw)
    #print(cleaned)
    '''
    # clean data
    text_data["Cleaned text"] = text_data["Review text"].map(lambda x: clean_data(x))

    # count vectorize the data
    vectorized_data = vectorize(text_data)
    #print(vectorized_data.T.sort_values(by=0, ascending=False).head(10))
    #print(vectorized_data)

    # tfidf vectorize for clustering
    tfidf_data, terms = tfidf_vectorize(text_data)

    return vectorized_data, target_r1v, target_r1a, target_r2v, \
           target_r2a, target_avg_val, target_avg_ar, tfidf_data, text_data, terms

def part_three():
    # making the reviews as either positive or negative
    df = pd.read_csv("sampleReviewData_030822.csv")
    df = df.drop(['No.', 'No. 2', 'No. 3'], axis=1, inplace=False)
    df = df.rename({'Review text': 'review_text', 'Rater 1 - Valence': 'rater1_valence', 'Rater 1 - Arousal': 'rater1_arousal', 'Rater 2 - Valence':'rater2_valence', 'Rater 2 - Arousal': 'rater2_arousal'}, axis=1, inplace=False)
    df = df[df['rater1_valence'] != 5]
    df['Positively_Rated'] = np.where(df['rater1_valence'] > 5, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(df['review_text'],
                                                        df['Positively_Rated'],
                                                        random_state=0)


    a = df.Positively_Rated.value_counts()

    x = list(a.index)
    y = list(a)

    # look at lengths of reviews
    df['length'] = df['review_text'].apply(len)
    df['length'].plot(bins=100, kind='hist', title=" Character Length Distribution")
    plt.show()
    # look at distributions for rating scores
    sns.countplot(x=df['rater1_valence']).set(title='Distribution of Rater 1 Valence Score')
    plt.show()
    sns.countplot(x=df['rater1_arousal']).set(title='Distribution of Rater 1 Arousal Score')
    plt.show()
    sns.countplot(x=df['rater2_valence']).set(title='Distribution of Rater 2 Valence Score')
    plt.show()
    sns.countplot(x=df['rater2_arousal']).set(title='Distribution of Rater 2 Arousal Score')
    plt.show()

    # visualize frequency of rating types
    ax = plt.bar(x=x, height=y, width=0.5, color=['green', 'red'], tick_label=x)
    plt.title('Frequency of Rating Types')
    plt.bar_label(ax, labels=y)
    plt.xlabel('Type of Rating', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    green_patch = mpatches.Patch(color='green', label='Positive Ratings')
    red_patch = mpatches.Patch(color='red', label='Negative Ratings')
    plt.legend(handles=[green_patch, red_patch])
    plt.show()


    scores_df = pd.DataFrame(columns=['Model', 'Accuracy'])

    # logistic regression V1
    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))
    #print('AUC for logistic regression V1 for rater 1 valence: ', roc_auc_score(y_test, predictions))
    log_v1_acc = accuracy_score(y_test, predictions)
    scores_df = scores_df.append({'Model' : 'LR_V1', 'Accuracy' : log_v1_acc}, ignore_index = True)
    feature_names = np.array(vect.get_feature_names())
    sorted_coef_index = model.coef_[0].argsort()

    '''
    print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
    print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
    print("")
    print("-"*30)
    print("")
    '''

    # logistic regression V2
    vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))
    #print('AUC for logistic regression V2 for rater 1 valence: ', roc_auc_score(y_test, predictions))
    log_v2_acc = accuracy_score(y_test, predictions)
    scores_df = scores_df.append({'Model' : 'LR_V2', 'Accuracy' : log_v2_acc}, ignore_index = True)
    feature_names = np.array(vect.get_feature_names())
    sorted_coef_index = model.coef_[0].argsort()

    '''
    print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
    print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
    print("")
    print("-" * 30)
    print("")
    '''

    # Multinomial NB based off positive or negative review
    NB_classifier = MultinomialNB()
    NB_classifier.fit(X_train_vectorized, y_train)
    y_predict_test = NB_classifier.predict(X_test_vectorized)
    cm = confusion_matrix(y_test, y_predict_test)
    MNB_pos_neg_acc = accuracy_score(y_test, y_predict_test)
    scores_df = scores_df.append({'Model' : 'MNB_Pos_Neg', 'Accuracy' : MNB_pos_neg_acc}, ignore_index = True)
    '''
    print("Multinomial NB based off positive or negative review for rater 1 valence: ")
    #sns.heatmap(cm, annot=True)
    print(classification_report(y_test, y_predict_test))
    print("")
    print("-" * 30)
    print("")
    '''


    # ANN
    model = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=5)

    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)
    ann_acc = accuracy_score(y_test, predictions)
    scores_df = scores_df.append({'Model' : 'ANN', 'Accuracy' : ann_acc}, ignore_index = True)

    '''
    print("MLPClassifier for rater 1 valence: ")
    print(confusion_matrix(y_test, predictions))

    print(classification_report(y_test, predictions))
    print("")
    print("-" * 30)
    print("")
    '''
    '''
    # wordclouds
    positive = df[df['sentiment'] == 1]
    negative = df[df['sentiment'] == 0]
    sentences = positive['review_text'].tolist()
    sentences_as_one_string = ' '.join(sentences)
    plt.figure(figsize=(20, 20))
    plt.imshow(WordCloud().generate(sentences_as_one_string))
    sentences = negative['review_text'].tolist()
    sentences_as_one_string = ' '.join(sentences)

    plt.figure(figsize=(20, 20))
    plt.imshow(WordCloud().generate(sentences_as_one_string))
    '''
    return scores_df

# Main Portion
#----------------------------------------------------

vectorized_data, target_r1v, target_r1a, target_r2v, \
           target_r2a, target_avg_val, target_avg_ar, tfidf_data, cleaned_data, words = format_data("sampleReviewData_030822.csv")

test_size = 0.2
# split into training and test sets
x_train_r1v, x_test_r1v, y_train_r1v, y_test_r1v = train_test_split(vectorized_data, target_r1v, test_size=test_size, random_state=3)
x_train_r1a, x_test_r1a, y_train_r1a, y_test_r1a = train_test_split(vectorized_data, target_r1a, test_size=test_size, random_state=3)
x_train_r2v, x_test_r2v, y_train_r2v, y_test_r2v = train_test_split(vectorized_data, target_r2v, test_size=test_size, random_state=3)
x_train_r2a, x_test_r2a, y_train_r2a, y_test_r2a = train_test_split(vectorized_data, target_r2a, test_size=test_size, random_state=3)
x_train_avg_val, x_test_avg_val, y_train_avg_val, y_test_avg_val = train_test_split(vectorized_data, target_avg_val, test_size=test_size, random_state=3)
x_train_avg_ar, x_test_avg_ar, y_train_avg_ar, y_test_avg_ar = train_test_split(vectorized_data, target_avg_ar, test_size=test_size, random_state=3)


# cross validate the different models
#-------------------------------------------------------------------
# Naive Bayes - Out of these three it performed the best
# SDGC Classifier - Worse than NB
# SVM - Worse than NB

# CV model for r1v
# 80 - 20 split
# avg = 0.331875
# 60 - 40 split
# avg = 0.3316
# 50 - 50 split
# avg = 0.322

MBN_normal_acc = mean(cross_val_score(MultinomialNB(), x_train_r1v, y_train_r1v , cv=10))

# performed worse than MNB
sgdc_acc = mean(cross_val_score(SGDClassifier( max_iter=5), x_train_r1v, y_train_r1v , cv=10))


# SVM model r1v (cross validation takes too long for this one)
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(x_train_r1v, y_train_r1v)
predictions_svm = clf_svm.predict(x_test_r1v)
svm_acc = accuracy_score(predictions_svm, y_test_r1v)

'''
# CV model for r1a
# 80 - 20 split
# avg = 0.204375
# 60 - 40 split
# avg = 0.196
# 50 - 50 split
# avg = 0.202
print("Cross validation for rater 1 arousal NB model")
print(mean(cross_val_score(MultinomialNB(), x_train_r1a, y_train_r1a, cv=10)))
print("")
print("-"*30)
print("")


# CV model for r2v
# 80 - 20 split
# avg = 0.28
# 60 - 40 split
# avg = 0.269
# 50 - 50 split
# avg = 0.276
print("Cross validation for rater 2 valence NB model")
print(mean(cross_val_score(MultinomialNB(),x_train_r2v, y_train_r2v , cv=10)))
print("")
print("-"*30)
print("")


# CV model for r2a
# 80 - 20 split
# avg = 0.188
# 60 - 40 split
# avg = 0.2
# 50 - 50 split
# avg = 0.194
print("Cross validation for rater 2 arousal NB model")
print(mean(cross_val_score(MultinomialNB(), x_train_r2a, y_train_r2a, cv=10)))
print("")
print("-"*30)
print("")


# CV model for avg valence
# 80 - 20 split
# avg = 0.3337
# 60 - 40 split
# avg = 0.335
# 50 - 50 split
# avg = 0.319
print("Cross validation for average valence NB model")
print(mean(cross_val_score(MultinomialNB(), x_train_avg_val, y_train_avg_val, cv=10)))
print("")
print("-"*30)
print("")


# CV model for avg arousal
# 80 - 20 split
# avg = 0.2169
# 60 - 40 split
# avg = 0.1983
# 50 - 50 split
# avg = 0.209
print("Cross validation for average arousal NB model")
print(mean(cross_val_score(MultinomialNB(), x_train_avg_ar, y_train_avg_ar, cv=10)))
print("")
print("-"*30)
print("")

'''
# Part 3 Additional Modeling
#-------------------------------------------------------------

# all scores are for rater 1 valence
scores_acc = part_three()


# add the earlier scores to the dataframe
scores_acc = scores_acc.append({'Model': 'MNB_Normal', 'Accuracy': MBN_normal_acc}, ignore_index = True)
scores_acc = scores_acc.append({'Model': 'SVM', 'Accuracy': svm_acc}, ignore_index = True)
scores_acc = scores_acc.append({'Model': 'SGDC', 'Accuracy': sgdc_acc}, ignore_index = True)
'''
ax = scores_acc.plot(x='Model', y='Accuracy', kind='bar', figsize=(12,7),
                color=['green', 'green', 'green', 'green'],
                rot=0)
ax.bar_label(ax.containers[0], label_type='edge')
ax.margins(y=0.2)
plt.ylim(0, 1)
plt.title("Accuracy Score for Models with Manipulated Ratings")
plt.xlabel("Model Type", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
green_patch = mpatches.Patch(color='green', label='Pos/Neg Review Labels')
#blue_patch = mpatches.Patch(color='blue', label='Regular Labels')
plt.legend(handles=[green_patch])
plt.show()
'''

# visualize all the scores
ax = scores_acc.plot(x='Model', y='Accuracy', kind='bar', figsize=(12,7),
                color=['green', 'green', 'green', 'green', 'blue', 'blue', 'blue'],
                rot=0)
ax.bar_label(ax.containers[0], label_type='edge')
ax.margins(y=0.2)
plt.title("Accuracy Score by Model")
plt.xlabel("Model Type", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
green_patch = mpatches.Patch(color='green', label='Pos/Neg Review Labels')
blue_patch = mpatches.Patch(color='blue', label='Regular Labels')
plt.legend(handles=[green_patch, blue_patch])
plt.show()

pd.set_option('display.max_columns', None)

# Clustering
#-------------------------------------------------------------
# elbow method to find k
# looks  like k=3
'''
Sum_SD = []
K = range(2,10)
for k in K:
    km = KMeans(n_clusters=k, max_iter=50, n_init=10, random_state = 3)
    km = km.fit(tfidf_data)
    Sum_SD.append(km.inertia_)

plt.plot(K, Sum_SD, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_SD')
plt.title('Elbow Method to find k')
plt.show()
'''
optimal_k = 3
cluster_mdl = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=100, n_init=10, random_state=3)
cluster_mdl.fit(tfidf_data)
labels = cluster_mdl.labels_

# do PCA simply to visualize the results
pca = PCA(n_components=3, random_state=3)
pca_data = pca.fit_transform(tfidf_data.toarray())
x1 = pca_data[:, 0]
x2 = pca_data[:, 1]
x3 = pca_data[:, 2]


cleaned_data['x1'] = x1
cleaned_data['x2'] = x2
cleaned_data['x3'] = x3

cleaned_data['cluster'] = labels

print('Top 10 words per cluster:')

# get the top 10 words in each cluster
words_df = pd.DataFrame(tfidf_data.todense()).groupby(labels).mean()
for i, r in words_df.iterrows():
    print('\nCluster {}'.format(i))
    print(','.join([words[w] for w in np.argsort(r)[-10:]]))


# visualize clusters
plt.figure(figsize=(12,7))
plt.title('KMeans Clustering')
plt.xlabel('x1')
plt.ylabel('x2')
sns.scatterplot(data=cleaned_data, x='x1', y='x2', hue='cluster', palette='viridis')
plt.show()

plt.figure(figsize=(12,7))
plt.title('KMeans Clustering')
ax = plt.axes(projection="3d")
ax.scatter3D(x1, x2, x3, c=cleaned_data['cluster'])
plt.show()


# Anomaly Detection
#---------------------------------------------------
# In Class method, KNN
# 1 = outlier
knn_mdl = KNN(n_neighbors=4, contamination=0.02)
knn_mdl.fit(vectorized_data)
cleaned_data["KNN Labels"] = knn_mdl.labels_

'''
print(cleaned_data["KNN Labels"].value_counts())
print("")
print("")
'''

# print out the rows which are considered outliers for KNN
KNN_outliers = cleaned_data.loc[cleaned_data["KNN Labels"] == 1]
# to view the records that are viewed as outliers for KNN
# uncomment the code below

'''
print("Printing the rows which are considered to be outliers for only KNN:")
print("-"*30)
print(KNN_outliers)
print("")
print("")
'''

# Out of Class, Isolation Forest
# contamination is default "auto"
# -1 = outlier
# to view the records that are viewed as outliers for Iso Forest auto
# uncomment the code below
# This identifies zero outliers in the dataset though

iso_mdl = IsolationForest(random_state=3)
cleaned_data["Isolation Forest Labels"] = iso_mdl.fit_predict(vectorized_data)

'''
print(cleaned_data["Isolation Forest Labels"].value_counts())
print("")
print("")

# print out the rows which are considered outliers for Iso Forest auto
iso_outliers = cleaned_data.loc[cleaned_data["Isolation Forest Labels"] == -1]
print("Printing the rows which are considered to be outliers for only Iso Forest:")
print("-"*30)
print(iso_outliers)
print("")
print("")
'''

# Out of Class, Isolation Forest
# contamination is same as KNN, "0.02"
# -1 = outlier
iso_mdl_2 = IsolationForest(random_state=3, contamination=0.02)
cleaned_data["Isolation Forest Labels 0.02"] = iso_mdl_2.fit_predict(vectorized_data)

'''
print(cleaned_data["Isolation Forest Labels 0.02"].value_counts())
print("")
print("")
'''

# print out the rows which are considered outliers for Iso Forest 0.02
iso_outliers_2 = cleaned_data.loc[cleaned_data["Isolation Forest Labels 0.02"] == -1]
# to view the records that are viewed as outliers for Iso Forest 0.02
# uncomment the code below

'''
print("Printing the rows which are considered to be outliers for only Iso Forest 0.02:")
print("-"*30)
print(iso_outliers_2)
print("")
print("")
'''

# print out the rows which are considered outliers for both KNN and Iso Forest 0.02
cleaned_data["Outliers KNN and Iso"] = cleaned_data.apply(lambda row: 1 if row["Isolation Forest Labels 0.02"] == -1 and row["KNN Labels"] == 1 else 0, axis=1)

combined_outliers = cleaned_data.loc[cleaned_data["Outliers KNN and Iso"] == 1]
print("Printing the rows which are considered to be outliers for both KNN and Iso Forest 0.02:")
print("-"*30)
print(combined_outliers)
print("")
print("")


# Visualize the outliers for each algorithm
# KNN outliers
fig = plt.figure(figsize=(12,7))
plt.title('KNN Anomaly Detection (PCA to visualize)')
ax = plt.axes(projection="3d")
ax.scatter3D(x1, x2, x3, c=cleaned_data['KNN Labels'], cmap='bwr')
plt.show()

# to view the graph that are viewed as outliers for Iso Forest auto
# uncomment the code below

# Iso Forest auto outliers
fig = plt.figure(figsize=(12,7))
plt.title('Isolation Forest Anomaly Detection (auto) ( (PCA to visualize)')
ax = plt.axes(projection="3d")
ax.scatter3D(x1, x2, x3, c=cleaned_data['Isolation Forest Labels'], cmap='bwr')
plt.show()

# Iso Forest 0.02 outliers
fig = plt.figure(figsize=(12,7))
plt.title('Isolation Forest Anomaly Detection (0.02) (PCA to visualize)')
ax = plt.axes(projection="3d")
ax.scatter3D(x1, x2, x3, c=cleaned_data['Isolation Forest Labels 0.02'], cmap='RdYlBu')
plt.show()

# Combined KNN and Iso Forest 0.02 outliers
plt.figure(figsize=(12,7))
plt.title('KNN and Iso Forest 0.02 Anomaly Detection (PCA to visualize)')
ax = plt.axes(projection="3d")
ax.scatter3D(x1, x2, x3, c=cleaned_data['Outliers KNN and Iso'], cmap='bwr')
plt.show()



