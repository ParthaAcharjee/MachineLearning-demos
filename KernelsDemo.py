import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm

plt.close('all')

def plotclasses(x,y,plottitle='Plot classes on feature domain'):
    ## Plot data
    fig=plt.figure()
    if np.shape(x)[1]==2:
        plt.scatter(x[y,0],x[y,1],marker='.', color='r',alpha=0.5,label='Red Class')
        plt.scatter(x[~y,0],x[~y,1],marker='.', color='g',alpha=0.5, label='Green Class')
    elif np.shape(x)[1]==3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[y,0],x[y,1],zs=x[y,2],marker='.', color='r',alpha=0.5,label='Red Class')
        ax.scatter(x[~y,0],x[~y,1],zs=x[~y,2],marker='.', color='g',alpha=0.5, label='Green Class')
        ax.set_zlabel('X_3')
        
    
    
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.title(plottitle)
    plt.legend()
    plt.show()



# Generate data
N=2000;
x=np.random.uniform(-1,1,(N,2))
y=np.sqrt(x[:,0]**2+x[:,1]**2)<0.5
plotclasses(x,y)


## SVM classification

clf = svm.LinearSVC()
clf.fit(x, y)
yest=clf.predict(x)
plotclasses(x,yest,'Linear SVM results')



# Manually introduce a kernel
X=np.zeros((N,3))
X[:,0:2]=x
X[:,2]=(x[:,0]**2+x[:,1]**2)**0.5

clf2 = svm.LinearSVC()
clf2.fit(X, y)
Yest=clf2.predict(X)
plotclasses(X,Yest,'Manual kernel results')

# Using RBF kernels

clf3 = svm.SVC()
clf3.fit(x, y)
yrbf=clf3.predict(x)
plotclasses(x,yrbf,'RBF kernel results')


clf4 = svm.SVC(kernel='poly',degree=2)
clf4.fit(x, y)
ypoly=clf4.predict(x)
plotclasses(x,ypoly,'Poly kernel results with degree 2')

# Plotting decision regions
#plt.figure()
#x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
#y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                     np.arange(y_min, y_max, 0.01))
#
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
#
#
#
#
#plt.contourf(xx, yy, Z, alpha=0.4,label='SVM boundaries')


