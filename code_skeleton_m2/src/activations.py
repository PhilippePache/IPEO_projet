import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        return 1/(1+np.exp(-z))

    @staticmethod
    def gradient(z):
        return Sigmoid.forward(z)*(1-Sigmoid.forward(z))

class ReLU:
    @staticmethod
    def forward(z):
        return 0*(z < 0) + z*(z >= 0)

    @staticmethod
    def gradient(z):
        return 0*(z < 0) + 1*(z >= 0)
    
class Softmax:
    @staticmethod
    def forward(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def gradient(z):
        return np.ones_like(z)

class Identity:
    @staticmethod
    def forward(z): return z
    
    @staticmethod
    def gradient(z): return np.ones_like(z)
