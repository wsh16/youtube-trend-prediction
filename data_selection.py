# %%

"""
This script can be used as skelton code to read the challenge train and test
csvs, to train a trivial model, and write data to the submission file.
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

# %%

from sklearn.metrics import accuracy_score

## Read csvs
train_df = pd.read_csv('youtube-new/USvideos.csv', index_col=0)

# %%

## Handle missing values
train_df.fillna('NA', inplace=True)

data_X = train_df[['trending_date', 'title', 'channel_title',
                   'category_id', 'publish_time', 'tags', 'likes',
                   'dislikes', 'comment_count', 'thumbnail_link', 'comments_disabled',
                   'ratings_disabled', 'video_error_or_removed', 'description']]

data_y = train_df[['views']]
# %%

## Filtering column "mail_type"
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, train_size=0.9)
# sns.distplot(int(train_df['mail_type']), kde=False)

tags = X_train['tags']
tags_test = X_test['tags']

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words="english")

## let's get counts for the first 5 tweets in the data
example_train_vectors = count_vectorizer.fit_transform(tags)

train_vectors = count_vectorizer.transform(tags)
test_vectors = count_vectorizer.transform(tags_test)

reg = LinearRegression().fit(train_vectors, y_train)
# reg = linear_model.Ridge(alpha=.5).fit(train_vectors, y_train)

result = reg.predict(test_vectors)

print(reg.score(test_vectors, y_test))

result_df = pd.DataFrame(result, columns=['views'])
result_df.to_csv("predict.csv", index=True)

y_test_df = pd.DataFrame(y_test, columns=['views'])
y_test_df.to_csv("test.csv", index=True)

# print(tags)
# print(len(tags))
#
# print(example_train_vectors[0].todense().shape)
# print(example_train_vectors[0].todense())


# %%

## PCA
# data_train = np.array(train_df.iloc[:, [3,4,6,7,8,9,10,11]])
# data_test = np.array(test_df.iloc[:, [3,4,6,7,8,9,10,11]])
# data = np.vstack((data_train, data_test))#0:10744是test
#
# M = np.mean(data, 0) # compute the mean
# Var = np.var(data,0)
# # C = data-M
# C = (data - M)*1/Var
# W = np.dot(C.T, C) # compute covariance matrix
# eigval, eigvec = np.linalg.eig(W) # compute eigenvalues and eigenvectors of covariance matrix
# idx = eigval.argsort()[::-1] # Sort eigenvalues
# eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
# newData2 = np.dot(C,np.real(eigvec[:,:2])) # Project the data to the new space (2-D)
# newData3 = np.dot(C,np.real(eigvec[:,:8])) # Project the data to the new space (3-D)
#
# newData2_train = newData2[0:25066]
# newData2_test = newData2[25066:35811]
#
# newData3_train = newData3[0:25066]
# newData3_test = newData3[25066:35811]


# newData3 = np.dot(C,np.real(eigvec[:,:3])) # Project the data to the new space (3-D)


# %%

## Do one hot encoding of categorical feature
# feat_enc = OneHotEncoder()
# feat_enc.fit(np.vstack((train_x, test_x)))
# train_x_featurized = feat_enc.transform(train_x)
# test_x_featurized = feat_enc.transform(test_x)
#
# train_type_array = train_x_featurized.A
# test_type_array = test_x_featurized.A
#
# data_type = np.vstack((train_type_array, test_type_array))
#
# M0 = np.mean(data_type, 0) # compute the mean
# Var0 = np.var(data_type,0)
# C0 = (data_type - M0)*1/Var0
#
# train_type_array = C0[0:25066]
# test_type_array = C0[25066:35811]
# # train_type_array = data_type[0:25066]
# # test_type_array = data_type[25066:35811]
#
# newData3_train_fin = np.hstack((newData3_train, train_type_array))
# newData3_test_fin = np.hstack((newData3_test, test_type_array))

# type(train_type_array = train_x_featurized.A)
