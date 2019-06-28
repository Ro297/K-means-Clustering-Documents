from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import glob
import re
import matplotlib as plt
from joblib import dump, load


def text_to_list(folderpath):
    x = []
    folder = glob.glob(folderpath)
    for file in folder:
        f = open(file,"r+")
        intro = f.read()
        x.append(intro)

    return x

relevant = text_to_list("Train - Relevant\\*.txt")
not_relevant = text_to_list("Train - Not Relevant\\*.txt")
documents = relevant + not_relevant


docs = []
for t in documents:
    no_num_t = no_number_preprocessor(t)
    docs.append(no_num_t)


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(docs)

true_k = 7
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=1)
model.fit(X)

dump(model, 'filename1.joblib') 


print(model.labels_[:71])
print((model.labels_[71:]))  

f = open('3.txt',"w")
xxx = model.labels_[71:]
for o in xxx:
    f.write(str(o))
    f.write('\n')
f.close()

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")


model = load('filename3.joblib') 

folder = glob.glob("Test\\*.txt")

pred = []
for file in folder:
    j = []
    test = open(file,"r+")
    i = test.read()
    j.append(i)

    Y = vectorizer.transform(j)
    prediction = model.predict(Y)
    pred.append(prediction[0])

f = open('2.txt',"w")
for o in pred:
    f.write(str(o))
    f.write('\n')
f.close()

print(pred)