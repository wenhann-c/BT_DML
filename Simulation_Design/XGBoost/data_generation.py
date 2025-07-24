import numpy as np

mean = np.linspace(0.2, 0.8, 4)
cov = np.array([[round(0.2**abs(i-j)*((-1.01)**(i+j)), 3) for j in range(4)] for i in range(4)])
beta = -4.0
df = 6

def g(x):
    if x.ndim == 1:
        x= x.reshape(1,-1)
    return 2*np.abs(x[:,0]*x[:,4])**(3/2)  - np.exp(x[:,1]*x[:,5]) + 3*x[:,2]**2 + np.sin(x[:,2]*x[:,3])- x[:,0]*x[:,2]

def d(x):
    if x.ndim == 1:
        x= x.reshape(1,-1)
    return 0.5*np.exp(x[:,1]) + x[:,1]*x[:,2] - np.sin(x[:,5]**2)

def get_data(N, rng):
    x_normal = rng.multivariate_normal(mean=mean,cov=cov,size=N)
    x_uniform = rng.uniform(0,1,size=(N,2))
    x_data = np.concatenate((x_normal, x_uniform),axis=1)

    xi = rng.logistic(size=N)
    d_data = (d(x_data) + xi >= 0).astype(float)

    epsilon = rng.normal(scale=np.sqrt(1.5), size=N)
    y_data = beta*d_data + g(x_data) + epsilon

    return y_data, d_data, x_data