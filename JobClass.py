
"""
@author: atakan
"""
#Importing Libraries
import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
%matplotlib inline
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import seaborn as sn

#Importing data 
dataset = pd.read_csv("fake_job_postings.csv")
dataset.head()
dataset.info()
    #finding nulls
null_columns=dataset.columns[dataset.isnull().any()]
dataset[null_columns].isnull().sum()
#droping colmuns with high number of nulls
dataset = dataset.drop(["salary_range", "department"], axis=1)
#subsetting columns for nlp text classification 
text_data = dataset[["title","company_profile","description","requirements","benefits"]]



text = " ".join(review for review in dataset.title)
print ("There are {} words in the combination of all title.".format(len(text)))
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["Jr.", ".NET"])

# Generate a word cloud image
wordcloud = WordCloud(background_color="white",colormap="prism").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure( figsize=(15,15))
plt.axis("off")
plt.imshow(wordcloud)
#☻Importing Image
hat_mask = np.array(Image.open("worker_hat.png"))
hat_mask[hat_mask == 0] = 255
#Wordcloud with worker hat
wc = WordCloud(background_color="white", max_words=1000, mask=hat_mask,
               stopwords=stopwords, contour_width=3, contour_color='firebrick')
# Generate a wordcloud
wc.generate(text)
# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

#☻Importing Image
teacher = np.array(Image.open("english_teacher.png"))
teacher[teacher == 0] = 255
#Wordcloud with teacher
wcteacher = WordCloud(background_color="white", max_words=500, mask=teacher,
               stopwords=stopwords, contour_width=3, contour_color='firebrick')
# Generate a wordcloud
wcteacher.generate(text)
# show
plt.figure(figsize=[20,10])
plt.imshow(wcteacher, interpolation='bilinear')
plt.axis("off")
plt.show()

#Description wordcloud
text_data = text_data.drop(text_data.index[[17513]])

des = " ".join(des for des in text_data.description)
print ("There are {} words in the combination of all description.".format(len(des)))


# Generate a word cloud image
wordcloud = WordCloud(background_color="white",colormap="twilight_r").generate(des)
# the matplotlib way:
plt.figure( figsize=(15,15))
plt.imshow(wordcloud)
plt.axis("off")
#there are too many stopword therefore I will clean them
#Data prep 
#making corpus for descriptions of jobs with first 2000 people
corpus = []
for i in range(0, 2000):
    description = re.sub('[^a-zA-Z]', ' ', dataset['description'][i])
    description = description.lower()
    description = description.split()
    ps = PorterStemmer()
    description = [ps.stem(word) for word in description if not word in set(stopwords.words('english'))]
    description = ' '.join(description)
    corpus.append(description)

# Creating the Bag of Words model with description
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

#I will add more columns to bag of word model but I need to clean them first 
#Again I will take first 2000 people first 
corpus_title = []
for i in range(0, 2000):
    R_title = re.sub('[^a-zA-Z]', ' ', dataset['title'][i])
    R_title = R_title.lower()
    R_title = R_title.split()
    ps = PorterStemmer()
    R_title = [ps.stem(word) for word in R_title if not word in set(stopwords.words('english'))]
    R_title = ' '.join(R_title)
    corpus_title.append(R_title)

#title bag of word model
cv = CountVectorizer()
X2 = cv.fit_transform(corpus_title).toarray()

#Again I will take first 2000 people first 
corpus_company_profile = []
for i in range(0, 2000):
    R_company_profile = re.sub('[^a-zA-Z]', ' ', str(dataset['company_profile'][i]))
    R_company_profile = R_company_profile.lower()
    R_company_profile = R_company_profile.split()
    ps = PorterStemmer()
    R_company_profile = [ps.stem(word) for word in R_company_profile if not word in set(stopwords.words('english'))]
    R_company_profile = ' '.join(R_company_profile)
    corpus_company_profile.append(R_company_profile)

#title bag of word model
cv = CountVectorizer()
X3 = cv.fit_transform(corpus_company_profile).toarray()

#Adding array to eachother for each person 
X_df = pd.DataFrame(X)
X2_df = pd.DataFrame(X2)
X3_df = pd.DataFrame(X3)
X_total = pd.concat([X_df, X2_df,X3_df], axis=1)
X_total_np = X_total.values

#Spliting set to test and training set 
from sklearn.model_selection import train_test_split
y = dataset.iloc[:,-1].values
y_1 = y[0:2000]
X_train, X_test, y_train, y_test = train_test_split(X_total_np, y_1, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#plotting confusion matrix
#p1
sn.set(font_scale=1.4) # for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size
plt.show()
#p2
sn.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')
plt.show()

#p3
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sn.heatmap(cm, annot=labels, fmt="", cmap='Blues')
plt.title("Confusion Matrix of Small Sample")

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

accuracy(cm)

#New Word Clouds(cleaned data) 
#Title image
corpus_tit_df = pd.DataFrame(corpus_title, columns=["title"])
text_title = " ".join(title for title in corpus_tit_df.title)
print ("There are {} words in the combination of all title.".format(len(text_title)))
stopword = set(STOPWORDS)
stopword.update(["manag"])

# Generate a word cloud image
wordcloud_title = WordCloud(background_color="white",colormap="prism").generate(text_title)

# Display the generated image:
# the matplotlib way:

plt.figure( figsize=(15,15))
plt.axis("off")
plt.imshow(wordcloud_title)

#Description image
corpus_des_df = pd.DataFrame(corpus, columns=["description"])
text_description = " ".join(description for description in corpus_des_df.description)
print ("There are {} words in the combination of all description.".format(len(text_description)))

# Generate a word cloud image
wordcloud_description = WordCloud(background_color="white",colormap="twilight_r").generate(text_description)

# Display the generated image:
# the matplotlib way:

plt.figure( figsize=(15,15))
plt.axis("off")
plt.imshow(wordcloud_description)




# 98% accuracy occured but now I will change my sample 
corpus_new_description = []
corpus_new_title = []
corpus_new_company_profile = []
for i in range(0, 6000):
    description = re.sub('[^a-zA-Z]', ' ', dataset['description'][i])
    description = description.lower()
    description = description.split()
    ps = PorterStemmer()
    description = [ps.stem(word) for word in description if not word in set(stopwords.words('english'))]
    description = ' '.join(description)
    corpus_new_description.append(description)
    R_title = re.sub('[^a-zA-Z]', ' ', dataset['title'][i])
    R_title = R_title.lower()
    R_title = R_title.split()
    ps = PorterStemmer()
    R_title = [ps.stem(word) for word in R_title if not word in set(stopwords.words('english'))]
    R_title = ' '.join(R_title)
    corpus_new_title.append(R_title)
    R_company_profile = re.sub('[^a-zA-Z]', ' ', str(dataset['company_profile'][i]))
    R_company_profile = R_company_profile.lower()
    R_company_profile = R_company_profile.split()
    ps = PorterStemmer()
    R_company_profile = [ps.stem(word) for word in R_company_profile if not word in set(stopwords.words('english'))]
    R_company_profile = ' '.join(R_company_profile)
    corpus_new_company_profile.append(R_company_profile)



#converting to bag of word models 
X5 = cv.fit_transform(corpus_new_description).toarray()
X6 = cv.fit_transform(corpus_new_title).toarray()
X7 = cv.fit_transform(corpus_new_company_profile).toarray()
X5_df = pd.DataFrame(X5)
X6_df = pd.DataFrame(X6)
X7_df = pd.DataFrame(X7)
X_total_big_sample = pd.concat([X5_df, X6_df,X7_df], axis=1)
X_total_big_sample_np = X_total_big_sample.values


#Spliting set to test and training set 
y_2 = y[0:6000]
X_train_big, X_test_big, y_train_big, y_test_big = train_test_split(X_total_big_sample_np, y_2, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(X_train_big, y_train_big)

# Predicting the Test set results
y_pred_big = classifier.predict(X_test_big)
# Making the Confusion Matrix
cm_big = confusion_matrix(y_test_big, y_pred_big)
print(cm_big)
accuracy(cm_big)

#cm plot with big data 
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in
                cm_big.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm_big.flatten()/np.sum(cm_big)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sn.heatmap(cm_big, annot=labels, fmt="", cmap='Blues')
plt.title("Confusion Matrix of Large Sample")


#only title prediction
#need to make 1 big cv to transform

cv = CountVectorizer(max_features = 1000)
x_single_pred = cv.fit_transform(corpus_new_title).toarray()

y_2 = y[0:6000]
X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(x_single_pred, y_2, test_size = 0.20, random_state = 0)
classifier_title = GaussianNB()
classifier_title.fit(X_train_title, y_train_title)

def TitlePredict(job):
    new_review = job
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = classifier_title.predict(new_X_test)
    print(new_y_pred)
TitlePredict("daily money team representative")
TitlePredict("ACANCY ASSISTANT ADMIN – COMPANY XYZ – US $ 17 / HOUR")



