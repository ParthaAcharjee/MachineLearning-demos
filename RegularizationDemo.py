import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# create dataset for a semi-circle
N=1000
x=np.linspace(-1,1,N)
y=np.sqrt(1-x**2)+np.random.normal(0,0.1,N)

plt.close('all')

### Linear regrassion fiting
model=10
X=np.zeros((N,model))
for i in range(1,model):
    X[:,i-1]=pow(x,i)

modelLinear = linear_model.LinearRegression()
modelLinear.fit(X,y)
coefsLinear=modelLinear.coef_
errorLinear=np.mean((y-modelLinear.predict(X))**2)

## L2 regularization error and coefficient relation

n_alphas = 200
alphas = np.logspace(-10, 1, n_alphas)

#alphas = [1e-100,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
#n_alphas=len(alphas)

modelL2 = linear_model.Ridge(fit_intercept=False)


coefsL2 = np.zeros((n_alphas,model))
errorL2 = np.zeros(n_alphas)
for a,k in zip(alphas,range(0,n_alphas)):
    modelL2.set_params(alpha=a)
    modelL2.fit(X, y)
    coefsL2[k,:]=modelL2.coef_
    errorL2[k]=np.mean((y-modelL2.predict(X))**2)
 

## L1 regularization and coefficeint relation

modelL1 = linear_model.Lasso(fit_intercept=False)

coefsL1 = np.zeros((n_alphas,model))
errorL1 = np.zeros(n_alphas)
for a,k in zip(alphas,range(0,n_alphas)):
    modelL1.set_params(alpha=a)
    modelL1.fit(X, y)
    coefsL1[k,:]=modelL1.coef_
    errorL1[k]=np.mean((y-modelL1.predict(X))**2)


## Plot coefficients weights for different alphas
fig,subplot=plt.subplots(2,1,sharex=True)

ax =subplot[0]

ax.plot(alphas, coefsL2)
ax.set_xscale('log')
ax.set_title('L2 coefficients as a function of the regularization')
ax.set_ylabel('weights')

ax=subplot[1]

ax.plot(alphas, coefsL1,'--')
ax.set_xscale('log')
ax.set_xlabel('alpha')
ax.set_ylabel('weights')

ax.set_title('L1 coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

## Plot Regularization (alpha) vs Error
plt.figure()
plt.plot(alphas,errorL1,'r',alphas,errorL2,'b')
plt.xscale('log')
plt.xlabel('Alphas')
plt.ylabel('Error')
plt.title('Regularization (alpha) vs Error')
plt.show()

## Plot Regularization (alpha) vs Coefficients==0
plt.figure()
L1zeros=np.sum(abs(coefsL1)<1e-15,axis=1)
L2zeros=np.sum(abs(coefsL2)<1e-15,axis=1)
plt.plot(alphas,L1zeros,'r',alphas,L2zeros,'b')
plt.xscale('log')
plt.xlabel('Alphas')
plt.ylabel('Number of zeros')
plt.title('Regularization (alpha) vs Coefficients==0')
plt.show()