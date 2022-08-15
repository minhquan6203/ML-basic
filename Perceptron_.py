from turtle import update
import numpy as np

class Perceptron:
    def __init__(self,learning_rate=0.01,n_iters=1000):
        self.lr=learning_rate
        self.iters=n_iters
        self.activation_func=self.unit_step
        self.weights=None
        self.bias=None


    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        y_=np.array([1 if i>0 else 0 for i in y])

        for i in range(self.iters):
            for idx, x_i in enumerate(X):
                output=np.dot(x_i,self.weights) + self.bias
                y_pred=self.activation_func(output)
                update=self.lr*(y_[idx]-y_pred)
                self.weights+=update*x_i
                self.bias+=update


    def predict(self,X):
        output=np.dot(X,self.weights)+self.bias
        y_pred=self.activation_func(output)
        return y_pred


    def unit_step(self,x):
        return np.where(x>=0,1,0)


