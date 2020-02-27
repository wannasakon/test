from django.shortcuts import render
import numpy as np
from sklearn import datasets
from sklearn import metrics
# Create your views here.
def index(req):
    group = ''
    submit = 'สแดงผล'
    if req.method == 'POST':
        print('เขา POST มา')
        group = (req.POST['group'])    
        
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups() #data.target_names
        categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
        train = fetch_20newsgroups(subset='train', categories=categories)
        test = fetch_20newsgroups(subset='test', categories=categories)
        len(train.data), len(test.data) #print(train.data[23])#print(train.data[3])
        print(train.target_names[train.target[3]])
    

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import make_pipeline
    
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(train.data, train.target)
    
        labels = model.predict(test.data)
        labels[0:10]
        test.target[0:10]
        n = len(test.data)
        corrects = [ 1 for i in range(n) if test.target[i] == labels[i] ]
        print(sum(corrects))
        sum(corrects)*100/n
    
        from sklearn.metrics import confusion_matrix
        mat = confusion_matrix(test.target, labels)
        # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Greens', xticklabels=train.target_names, yticklabels=train.target_names)
        # plt.xlabel('true label')
        # plt.ylabel('predicted label');
        def predict_category(s, train=train, model=model):
            pred = model.predict([s])
            return train.target_names[pred[0]]
        predict_category(group)    
    return render(req, 'myapp/index.html',{ 
        'result': submit,
        'group': group, 
    })

