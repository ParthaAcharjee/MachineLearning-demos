import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

plt.close('all')

# Generate data
N=500;
x=np.random.uniform(-1,1,(N,2))
y=np.sqrt(x[:,0]**2+x[:,1]**2)<0.5


xtest=np.random.uniform(-1,1,(N,2))
ytest=np.sqrt(xtest[:,0]**2+xtest[:,1]**2)<0.5

## Plot data
plt.scatter(x[y,0],x[y,1],marker='.', color='r',alpha=0.5,label='Red Class')
plt.scatter(x[~y,0],x[~y,1],marker='.', color='g',alpha=0.5, label='Green Class')

plt.xlabel('X_1')
plt.ylabel('X_2')
plt.title('Class map in feature dimension')
plt.legend()
plt.show()

# SVM classification

clf = svm.SVC()
clf.fit(x, y)
yest=clf.predict(xtest)


# Plotting decision regions
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)



plt.figure()
plt.contourf(xx, yy, Z, alpha=0.4,label='SVM boundaries')
plt.scatter(xtest[yest,0],xtest[yest,1],marker='.', color='r',alpha=0.5,label='Red Testset')
plt.scatter(xtest[~yest,0],xtest[~yest,1],marker='.', color='g',alpha=0.5, label='Green Testset')

plt.xlabel('X_1')
plt.ylabel('X_2')
plt.title('Class map in feature dimension')
plt.legend()
plt.show()

