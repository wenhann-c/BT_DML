import numpy as np

mean = np.array([1.0,2.0])
cov = np.array([[0.8,-0.5],[-0.5,1]])
beta = -2

def g(x):
    if x.ndim == 1:
        x= x.reshape(1,-1)
    return 2*x[:,0] - x[:,1]**2 + 2*x[:,2]**0.5 + np.exp(x[:,3])

def d(x):
    if x.ndim == 1:
        x= x.reshape(1,-1)
    return 0.5*x[:,1] + np.log(x[:,3])

def get_data(N, rng):
    x_normal = rng.multivariate_normal(mean=mean,cov=cov,size=N)
    x_uniform = rng.uniform(size=(N,2))
    x_data = np.concatenate((x_normal, x_uniform),axis=1)

    xi = rng.normal(scale=2)
    d_data = d(x_data) + xi

    epsilon = rng.normal(scale=x_data[:,2])
    y_data = beta*d_data + g(x_data) + epsilon

    return y_data, d_data, x_data