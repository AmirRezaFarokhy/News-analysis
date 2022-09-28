import pandas as pd
from collections import Counter, deque
import re
import numpy as np
import random

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

MAX_LEN = 12
VALID_LEN_POS = 750
VALID_LEN_NEU = 2270
LIST_NAME = ['positive', 'negetive', 'neutral']


df = pd.read_csv("all-data.csv", encoding="latin1")

# target updating
def classify(data):
    if data=="positive":
        return 1
    elif data=="negative":
        return 2
    elif data=="neutral":
        return 3


def PrepareData():
    lst = []
    for t in df['News'].values:
        texts = t.lower()
        texts = re.sub("[%;,.]+", repl="", string=texts)
        #texts = "<start> "  + texts + "<end>"
        lst.append(texts)


    strings_all = ''.join(i for i in lst)
    strings_all = re.sub(' +', repl=' ', string=strings_all)
    strings_all = Counter(strings_all.split(' '))

    string_each = []
    for l in lst:
        texts = re.sub(' +', repl=' ', string=l)
        each = texts.split(' ')
        string_each.append(Counter(each[:-1]))
    
    return string_each, strings_all


# Tokenization with the formula available in the README.md file 
def Tokenizer(each, group):
    value = []
    for count in each:
        val = []
        for k in count:
            if group[k]!=0:
                point_each = count[k] / len(count)
                point_all = np.log(len(each) / group[k])
                val.append(point_each*point_all)
                
        value.append(val)
        
    datasets = []
    for target, features in zip(df['target'], value):
        features = features[:MAX_LEN]
        if len(features)<MAX_LEN:
            mean = np.mean(features)
            length = MAX_LEN - len(features)
            for i in range(length):
                features.append(mean)
            datasets.append([target, features])
    
    # exiting the sort mode 
    random.shuffle(datasets)
    
    return datasets
       

df["target"] = list(map(classify, df["target"]))

# we must sort to can balace data
df.sort_values('target', inplace=True)

# data must be balanced
df = df[VALID_LEN_POS:]
df = df[:-VALID_LEN_NEU]

string_each, strings_all = PrepareData()
dataset = Tokenizer(string_each, strings_all)

X = []
y = []
for label, feature in dataset:
    X.append(feature)
    y.append(label)
    

X = np.array(X, dtype=np.float64)
y = np.array(y)

# Scaling data 
scaler = StandardScaler()
X = scaler.fit_transform(X)

#X = scale(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = SVC(kernel="rbf")
model.fit(x_train, y_train)

print(f"accuracy for this algoritm {model.score(x_test, y_test)}")

# show some sample predict
for real, pred in zip(y_test, model.predict(x_test)):
    print(f"real is {LIST_NAME[real-1]}, prediction is {LIST_NAME[real-1]}, \n")


