from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


mnist=fetch_openml('mnist_784')
# loading mnist datasets into x and y
x,y=mnist["data"],mnist["target"]

# this is done so that the data is split for training and testing
x_train=x[0:60000]
x_test = x[60000:70000]
y_train = y[0:60000]
y_test = y[60000:70000]
digit_show=x[65432]

#reshaping the selected digit into 28p*28px format
digit_img=digit_show.reshape(28,28) 

# this is done so that the system is trained with all the datasets
shuffle=np.random.permutation(60000)
x_train=x[shuffle]
y_train = y[shuffle]


# 6 detector this will give a binary output ie---TRUE/FALSE. so we are making a binary classifier
y_train=y_train.astype(np.int8)
y_test = y_test.astype(np.int8) #these lines are used to convert the string to int in numpy
y_train_6=(y_train==5)
y_test_6 = (y_test == 5)





# this line is used to plot the given select digit in matplot lib
plt.imshow(digit_img,cmap=matplotlib.cm.binary,interpolation="nearest")

# adding a classifier
clf=LogisticRegression(tol=0.1)
clf.fit(x_train,y_train_6)
a=clf.predict([digit_show])
print("\n\n\nThis is ", a)
# cross_val_score(clf,x_train,y_test_6,cv=3,scoring="accuracy")
plt.show()
