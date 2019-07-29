

import numpy as np
from matplotlib import pyplot as plt



#Input Data  [X values, Y values, Bias term ]
X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1,6,-1],
    [2,4,-1],
    [6,2,-1]
])

#Associaterd output labels
y = np.array([-1,-1,1,1,1])


for d, sample in enumerate(X):

    #Plot the negative sample(the first  2 )
    if d < 2:

        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidth=2)

    #Plot the positive example (the 3 left)
    else:

        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidth=2)



#Stochastic gradientes descent to learn the seperating hyperplan
def svm_sgd_plot(X,y):

    #initialize the SVMs wight vector with ceros (3 values)
    w = np.zeros(len(X[0]))

    #The learning rate
    eta = 1

    #This is the iterations to train
    epochs = 10000

    #store misclassification so we can lot the changes
    errors = []


    #gradient descent part
    for epochs in range(1,epochs):
        error = 0
        for i, x in enumerate(X):

            #missclassificatiom
            if(y[i]*np.dot([X[i]], w)) < 1:

                #missclassified update for our wieghts
                w = w + eta * (X[i]*y[i] +(-2 * (1/epochs)*w))
                error = 1
            else:
                #correct classification, update our weight

                w = w +eta * (-2(1/epochs) * w)
        errors.append(error)





    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()


    return w
