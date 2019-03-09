import sklearn
import cv2
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

ppc = 16
hog_images = []
hog_features = []
movie_img = []
movie_label = []
for dir_path in glob.glob("train/*"):
    img_label = dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(
                ppc, ppc), cells_per_block=(4, 4), block_norm='L2', visualize=True)
        hog_images.append(hog_image)
        hog_features.append(fd)

        movie_img.append(image)
        movie_label.append(img_label)

movie_img = np.array(movie_img)
movie_label = np.array(movie_label)
enc = preprocessing.LabelEncoder()
enc.fit(movie_label)
new_movie_label = enc.transform(movie_label)
print(new_movie_label)


hog_features = np.array(hog_features)
#movie_img = movie_img.reshape(1617, 32*32*3)
#print(movie_img.shape)
X_train, X_test, y_train, y_test = train_test_split(
    hog_features, new_movie_label, test_size=0.2, random_state=42, shuffle=True)

#SVM
clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
          max_iter=-1, probability=False, random_state=None, shrinking=True,
          tol=0.001, verbose=False)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
filename = 'SVM_model.sav'
joblib.dump(clf, filename)

# Random forest
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=2, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       oob_score=False, random_state=0, verbose=0, warm_start=False)
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))
filename = 'random_forest_model.sav'
joblib.dump(clf, filename)

