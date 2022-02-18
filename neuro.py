import numpy as np
import sklearn
from sklearn import svm

SVC_model=svm.SVC()

x=np.array([
    [0,0],
    [1,1],
    [0,1],
    [1,0],
])
y=np.array([
    1,
    1,
    0,
    0
])

for i in range(10000):
    SVC_model.fit(x,y)

v=np.array([[0,0]])#expect 1
print(SVC_model.predict(v))
