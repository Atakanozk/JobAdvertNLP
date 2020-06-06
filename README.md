Hi Folks! 
Today I will tell you about the NLP model I have created. 
The data I am working on contain descriptions of about 18000 job postings, titles, company descriptions and similar information.
The data also say whether the advertisements are real or not. 
Before creating the model, I cleaned the texts. 
Considering the titles, company information and job descriptions, I built an NLP model on those. According to those criteria, I found out whether the advertisements were real or not using naive Bayes classification. 
I prepared 2 different models by changing the sample size. The first model contains 2000 and the second model contains 6000 jobs adverts. 
The accuracy of the first model was 0.9875 and of the second one - 0.9733. Finally, I created a prediction function based on the job adverts’ titles. 
When you give the title of a job advert to Python, it says whether it is fake or real. 
If you have any suggestion or improvements, feel free to comment below.
TitlePredict("daily money team representative")
TitlePredict("ACANCY ASSISTANT ADMIN – COMPANY XYZ – US $ 17 / HOUR")
[1](Fake)
[0](Real)
You can find the data on the Kaggle website: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
