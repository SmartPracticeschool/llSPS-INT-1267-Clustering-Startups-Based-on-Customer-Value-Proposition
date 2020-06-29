
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import re
from keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
class Extract:
    # Place All As A Function For Reuseability
    def prediction(self,text):
        label = LabelEncoder()
        label.classes_= np.load('classes.npy',allow_pickle=True)
        m=load_model('cluster.h5')
        m.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        cv = pickle.load(open("vector.pickel", "rb"))
        ps = PorterStemmer()
        inp=text
        data=[]
        review=inp
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        data.append(review)    
        pred=cv.transform(data)
        pred_op = m.predict(pred)
        return label.classes_[list(pred_op[0]).index(max(pred_op[0]))]
o=Extract()
pickle.dump(o, open('model.pkl','wb'))
