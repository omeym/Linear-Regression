import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    r,c = np.shape(X)
    num_samples = r

    w_r = np.size(w, 0)

    if(c == w_r):
        predicted_value = np.matmul(X,w)
    else:
        predicted_value = np.matmul(X.transpose(),w)

    err = (1/num_samples) * np.sum(np.power((y-predicted_value),2))

    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  #####################################################		
  w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),y))
  
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################		
    X_mat = np.matmul(np.transpose(X),X)
    r,c = np.shape(X_mat)
    lambda_mat = lambd * np.identity(r)
    X_regularized = np.add(X_mat, lambda_mat)

    w = np.matmul(np.linalg.inv(X_regularized),np.matmul(np.transpose(X),y))

    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    lamdaset = np.geomspace(2**-14,1,15)
    bestMSE = float('inf')
    bestlambda = lamdaset[0]
   
    for lamda in lamdaset:
        w = regularized_linear_regression(Xtrain, ytrain, lamda)
        currentMSE = mean_square_error(w, Xval, yval)
        
        if(currentMSE < bestMSE):
            bestlambda = lamda
            bestMSE = currentMSE
        

    return bestlambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		
    X_1 = X
    for i in range(p):
        X = np.concatenate((X, np.power(X_1,(i+1))), axis=1)

    
    return X

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

