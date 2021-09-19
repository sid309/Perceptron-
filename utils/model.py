import numpy as np
from tqdm import tqdm
class Perceptron:
  def __init__(self,eta,epochs):
    self.eta=eta
    self.epochs=epochs
    self.weights=np.random.randn(3)*1e-4
  def activation_function(self,inputs,weights):
    z=np.dot(inputs,weights)
    return np.where(z>0,1,0)
  def fit(self,X,Y):
    self.X=X
    self.Y=Y
    X_with_bias=np.c_[self.X,-np.ones((len(self.X),1))]
    for epoch in tqdm(range(self.epochs),total=self.epochs,desc="training the model"):
      print("--"*10)
      print(f"for epoch: {epoch}")
      print("--"*10)
      y_hat=self.activation_function(X_with_bias,self.weights)
      print(f"predicted value after forward pass: \n{y_hat}")
      self.error=self.Y-y_hat
      print(f"error: \n{self.error}")
      self.weights=self.weights+self.eta*np.dot(X_with_bias.T,self.error)
      print(f"updated weights after epoch:\n{epoch}/{self.epochs} : \n{self.weights}")
      print("#####"*10)
  def predict(self,X):
      X_with_bias=np.c_[X,-np.ones((len(X),1))]
      return self.activation_function(X_with_bias,self.weights) 

  def total_loss(self):
      total_loss=np.sum(self.error)
      print(f"total loss: {total_loss}")
      return total_loss