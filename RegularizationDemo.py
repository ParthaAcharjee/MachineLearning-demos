import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
plt.close('all')
# create dataset for a semi-circle
N=1000
x=np.linspace(-1,1,N)
ymodel=np.sqrt(1-x**2)
ymodel=np.sin(2*np.pi*x)
y=ymodel+np.random.normal(0,0.001,N)
noise=np.zeros(N)
noise[10:N-10:20]=1;
y[noise==1]=5



### Linear regrassion fiting
model=50
X=np.zeros((N,model))
for i in range(1,model):
    X[:,i-1]=pow(x,i)

modelLinear = linear_model.LinearRegression()
modelLinear.fit(X,y)
coefsLinear=modelLinear.coef_
yestL0=modelLinear.predict(X)

## L2 regularization error and coefficient relation

n_alphas = 200
alphas = np.logspace(2, -5, n_alphas)


modelL2 = linear_model.Ridge(fit_intercept=False)


coefsL2 = np.zeros((n_alphas,model))
errorL2 = np.zeros(n_alphas)
modelDeviationL2=np.zeros(n_alphas)
for a,k in zip(alphas,range(0,n_alphas)):
    modelL2.set_params(alpha=a)
    modelL2.fit(X, y)
    coefsL2[k,:]=modelL2.coef_
    yest=modelL2.predict(X)
    errorL2[k]=np.mean((y-yest)**2)
    modelDeviationL2[k]=np.mean((ymodel-yest)**2)
 

## L1 regularization and coefficeint relation

modelL1 = linear_model.Lasso(fit_intercept=False)

coefsL1 = np.zeros((n_alphas,model))
errorL1 = np.zeros(n_alphas)
modelDeviationL1=np.zeros(n_alphas)

for a,k in zip(alphas,range(0,n_alphas)):
    modelL1.set_params(alpha=a)
    modelL1.fit(X, y)
    coefsL1[k,:]=modelL1.coef_
    yest=modelL1.predict(X)
    errorL1[k]=np.mean((y-yest)**2)
    modelDeviationL1[k]=np.mean((ymodel-yest)**2)


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
plt.plot(alphas,modelDeviationL1,color='r',ls='--')
plt.plot(alphas,modelDeviationL2,color='g',ls='--')
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

yestL1=modelL1.predict(X)
yestL2=modelL2.predict(X)
plt.figure()
plt.scatter(x,y,alpha=0.2)
plt.plot(x,ymodel,color='k')
plt.plot(x,yestL0, color='r',ls='--')
plt.plot(x,yestL1,color='g',ls='--')
plt.plot(x,yestL2,color='b',ls='--')
plt.show()

print("\nError from data L0: ",(np.sqrt(np.mean(y-yestL0)**2)))
print("Error from data L1: ",(np.sqrt(np.mean(y-yestL1)**2)))
print("Error from data L2: ",(np.sqrt(np.mean(y-yestL2)**2)))

print("\nError from model L0: ",(np.sqrt(np.mean(ymodel-yestL0)**2)))
print("Error from model L1: ",(np.sqrt(np.mean(ymodel-yestL1)**2)))
print("Error from model L2: ",(np.sqrt(np.mean(ymodel-yestL2)**2)))