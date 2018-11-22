import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from pandas import DataFrame

# Training set:
X, Y = make_blobs(n_samples=400, centers=2, n_features=2,cluster_std=1.5, center_box=(-4.0, 4.0),random_state=42)

## Add a bias column to the data
X = np.hstack([np.ones((X.shape[0], 1)),X])
Y = np.array(Y)
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)

#Activation function
def Sigmoid(z):
    return 1/(1 + np.exp(-z))

#The model to consider
def Hypothesis(w, x):
    return Sigmoid(x @ w)

#Cross entropy as cost function
def Cost_Function(X,Y,w,m):
    hi = Hypothesis(w, X)
    y = Y.reshape(-1, 1)
    J = 1/float(m) * np.sum(-y * np.log(hi) - (1-y) * np.log(1-hi))
    return J

#Derivative of cost function
def Cost_Function_Derivative(X,Y,w,m,alpha):
    hi = Hypothesis(w,X)
    y = Y.reshape(-1, 1)
    J = alpha/float(m) * X.T @ (hi - y)
    return J

#Batch Gradient descent method
def Gradient_Descent(X,Y,w,m,alpha):
    new_w = w - Cost_Function_Derivative(X,Y,w,m,alpha)
    return new_w

#Accuracy
def Accuracy(w):
    acc_unit = 0
    length = len(X)
    prediction = (Hypothesis(w, X) > 0.5)
    y = Y.reshape(-1, 1)
    acc_unit = prediction == y
    my_accuracy = (np.sum(acc_unit) / length)*100
    return my_accuracy
    #print('LR Accuracy %: ', my_accuracy)

def Logistic_Regression(X,Y,alpha,w1,w2,w3,num_iters):
    m = len(Y)
    accuracyw1 = []
    accuracyw2 = []
    accuracyw3 = []
    minibatch_size = 10

    for x in range(num_iters):

        #Batch Gradient Descent
        w1 = Gradient_Descent(X,Y,w1,m,alpha)
        accuracyw1.append(Accuracy(w1))

        #Stochastic Gradient Descent
        ideal_alpha = 3.2
        random_index = np.random.randint(m)
        Xnew = X[random_index:random_index + 1]
        Ynew = Y[random_index:random_index + 1]
        w2 = Gradient_Descent(Xnew,Ynew,w2,m,ideal_alpha)
        accuracyw2.append(Accuracy(w2))

        #Mini Batch Gradient Descent
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        Y_shuffled = Y[shuffled_indices]
        for i in range(0, m, minibatch_size):
          Xshuffled_new = X_shuffled[i:i + minibatch_size]
          Yshuffled_new = Y_shuffled[i:i + minibatch_size]
          gradients = float(m) / minibatch_size * Cost_Function_Derivative(Xshuffled_new,Yshuffled_new,w3,m,alpha)
          w3 = w3 - alpha * gradients
        accuracyw3.append(Accuracy(w3))

    #Plot the accuracy vs. iteration
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylabel('Accuracy %')
    ax.set_xlabel('Iterations')
    ax.plot(accuracyw1, c="black", label="Batch GD")
    ax.plot(accuracyw2, c="green", label="Stochastic GD")
    ax.plot(accuracyw3, c="red", label="MiniBatch GD")
    ax.legend(loc = 1, bbox_to_anchor=(0.6,0.5))

    #Plot the data
    df = DataFrame(dict(x=X[:, 1], y=X[:, 2], label=Y))
    colors = {0: 'red', 1: 'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

    #Plot the decision lines
    s = np.linspace(-5, 5, 100)
    Batch_GD = -w1[0]/w1[2] * s - w1[0]/w1[2]
    Stochastic_GD = -w2[0]/w2[2] * s - w2[0]/w2[2]
    Mini_Batch_GD = -w3[0]/w3[2] * s - w3[0]/w3[2]

    plt.plot(s, Batch_GD, '-', c="black", marker='.',linestyle=':',label="Batch GD")
    plt.plot(s, Stochastic_GD, '-', c="green", marker='.',linestyle=':',label="Stochastic GD")
    plt.plot(s, Mini_Batch_GD, '-', c="red", marker='.',linestyle=':',label="MiniBatch GD")
    ax.legend(loc=2, bbox_to_anchor=(0.8, 0.8))
    plt.show()

#Initiate the values
ep = .012
initial_w = np.random.rand(X.shape[1],1) * 2 * ep - ep
alpha = .1
iterations = 4000

Logistic_Regression(X,Y,alpha,initial_w,initial_w,initial_w,iterations)
