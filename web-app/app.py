import numpy as np
import flask
import pickle
import csv
import pandas as pd
from sklearn.metrics import pairwise_distances
from flask import Flask, render_template, request


app = Flask(__name__)

def bag_of_words_model(doc_id, num_results,to_append):
     myData = [["asin","brand","color","medium_image_url","product_type_name","title","formatted_price"],
                   ['B00IAA4JIQ','I Love Lucy','Purple','https://images-na.ssl-images-amazon.com/images/I/41cd6hd0eZL._SL160_.jpg','SHIRT',to_append,'$14.66']]
     myFile = open('data_new.csv', 'w')
     with myFile:
               writer = csv.writer(myFile)
               writer.writerows(myData)  
     print("Writing complete")
     df1=pd.read_pickle('afterduplicate_data.txt')
     df2=pd.read_csv('data_new.csv')
     result = pd.merge(df1,df2 , sort=False,how='outer')
     result.tail()
     from sklearn.feature_extraction.text import CountVectorizer
     title_vectorizer = CountVectorizer()
     title_features   = title_vectorizer.fit_transform(result['title'])
     title_features.get_shape()
     pairwise_dist = pairwise_distances(title_features,title_features[doc_id])
     indices = np.argsort(pairwise_dist.flatten())[0:num_results]
     df_indices = list(result.index[indices])
     recs=[]
     rec_list=[]
     url=[]
     for i in range(1,len(indices)):
        recs.append(result['title'].loc[df_indices[i]])
        rec_list.append(result['asin'].loc[df_indices[i]])
        url.append(result['medium_image_url'].loc[df_indices[i]])
     return recs, rec_list, url

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/rec')
def rec():
    return render_template('rec.html')

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
      if request.form["Title"]=="":
          return render_template('rec.html')
      else:
        to_append = request.form["Title"]  
        print(to_append)
        recs,rec_list,url = bag_of_words_model(4240, 6, to_append)
        filename=[]
        for i in range(0,len(rec_list)) :
                filename.append("http://www.amazon.com/dp/"+rec_list[i])
        return render_template('result.html',recs=recs,filename=filename, url = url, x=to_append)

if __name__ == '__main__':
    app.run(debug=True, port=8080)