import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv('dataset/Restaurant_Reviews.txt',delimiter='\t')
cv=CountVectorizer(stop_words='english')
X=cv.fit_transform(df.Review).todense()
y=df.iloc[:,1].values
mnb=MultinomialNB()
mnb.fit(X,y)
msg=input('enter msg:')
X_test=cv.transform([msg])
pred=mnb.predict(X_test)
print(pred[0])


