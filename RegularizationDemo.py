import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# create dataset for a semi-circle
N=1000
x=np.linspace(-1,1,N)
y=x**2+np.sqrt(1-x**2)+np.random.normal(0,0.1,N)

plt.close('all')

#### Calculate error and fiting using 1-9th polynomial
#f, ax = plt.subplots(3,3, sharex=True, sharey=True)
#for i in range(1,10):
#    z = np.polyfit(x, y, i)
#    
#    ithPoly=np.poly1d(z)
#    yest=ithPoly(x)
#    error=np.mean((yest-y)**2)
#
#    
#    plotid=(int(np.floor((i-1)/3)),(i-1)%3)
#    ax[plotid].scatter(x,y,marker='.',alpha=0.2)
#    ax[plotid].plot(x,yest, color='m')
#    ax[plotid].set_title('Fiting model. %d,'%i+'Error: %0.4f'%error)
#
#plt.show()
#
#### Linear regression error and coefficient relation
#M=50
#error=np.zeros(M)
#coef=np.zeros((M,M+1))
#for i in range(1,M+1):
#    z = np.polyfit(x, y, i)
#    
#    ithPoly=np.poly1d(z)
#    yest=ithPoly(x)
#    error[i-1]=np.mean((yest-y)**2)
#    coef[i-1,0:i+1]=ithPoly
#
#f,ax=plt.subplots(1,2)
#ax[0].imshow(abs(coef),interpolation='nearest'); ax[0].set_xlabel('Coeficient number'); ax[0].set_ylabel('Model number');
#ax[1].plot(error,marker='o',ls='--');  ax[1].set_xlabel('Model number'); ax[1].set_ylabel('Error')
#plt.show()

model=10
X=np.zeros((N,model))
for i in range(1,model):
    X[:,i-1]=pow(x,i)


fig,subplot=plt.subplots(2,1,sharex=True)

## L2 regularization error and coefficient relation

n_alphas = 200
alphas = np.logspace(-10, 1, n_alphas)
modelL2 = linear_model.Ridge(fit_intercept=False)


coefsL2 = np.zeros((n_alphas,model))
errorL2 = np.zeros(n_alphas)
for a,k in zip(alphas,range(0,n_alphas)):
    modelL2.set_params(alpha=a)
    modelL2.fit(X, y)
    coefsL2[k,:]=modelL2.coef_
    errorL2[k]=np.mean((y-modelL2.predict(X))**2)
 
ax =subplot[0]

ax.plot(alphas, coefsL2)
ax.set_xscale('log')
ax.set_title('L2 coefficients as a function of the regularization')
ax.set_ylabel('weights')


## L1 regularization and coefficeint relation

modelL1 = linear_model.Lasso(fit_intercept=False)

coefsL1 = np.zeros((n_alphas,model))
errorL1 = np.zeros(n_alphas)
for a,k in zip(alphas,range(0,n_alphas)):
    modelL1.set_params(alpha=a)
    modelL1.fit(X, y)
    coefsL1[k,:]=modelL1.coef_
    errorL1[k]=np.mean((y-modelL1.predict(X))**2)


ax=subplot[1]

ax.plot(alphas, coefsL1,'--')
ax.set_xscale('log')
ax.set_xlabel('alpha')
ax.set_ylabel('weights')


ax.set_title('L1 coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

plt.figure()
plt.plot(alphas,errorL1,'r',alphas,errorL2,'b')
plt.xscale('log')
plt.xlabel('Alphas')
plt.ylabel('Error')
plt.title('Regularization (alpha) vs Error')
plt.show()

plt.figure()
L1zeros=np.sum(coefsL1==0,axis=1)
L2zeros=np.sum(coefsL2==0,axis=1)
plt.plot(alphas,L1zeros,'r',alphas,L2zeros,'b')
plt.xscale('log')
plt.xlabel('Alphas')
plt.ylabel('Number of zeros')
plt.title('Regularization (alpha) vs Coefficients==0')
plt.show()