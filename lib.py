import scipy.linalg
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


def NeuralNetworkPass(x, w): #x is 3 x 1, w is 16 x 1 vectors; calculate the output of the assignment's neural network
    
    perceptronOneOutput = w[0] * np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]) #three peceptrons, each element of the input vector has a weight, then passed through tanh nonlinearity. output of nonlinearity also has a weight
    perceptronTwoOutput = w[5] * np.tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
    perceptronThreeOutput = w[10] * np.tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])

    output = perceptronOneOutput + perceptronTwoOutput + perceptronThreeOutput + w[15] #summation of the peceptron outputs with a weight at the final output

    return output

def CalculateNeuralNetworkGradient(x, w): #x is 3 x 1, w is 16 x 1 vectors
    gradientVector = np.zeros(16) #finding partial 16 partial derivatives (because of 16 weights) of neural network function; this is a tall vector

    #use chain rule

    gradientVector[0] = np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]) #weight vanishes since it is outside of tanh
    gradientVector[1] = w[0] * (1 - np.square((np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])))) * x[0] #use chain rule since weight is inside of tanh
    gradientVector[2] = w[0] * (1 - np.square((np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])))) * x[1] #derivative of tanh is 1 - tanh^2
    gradientVector[3] = w[0] * (1 - np.square((np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])))) * x[2]
    gradientVector[4] = w[0] * (1 - np.square((np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]))))

    gradientVector[5] = np.tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
    gradientVector[6] = w[5] * (1 - np.square((np.tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])))) * x[0]
    gradientVector[7] = w[5] * (1 - np.square((np.tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])))) * x[1]
    gradientVector[8] = w[5] * (1 - np.square((np.tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])))) * x[2]
    gradientVector[9] = w[5] * (1 - np.square((np.tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9]))))

    gradientVector[10] = np.tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])
    gradientVector[11] = w[10] * (1 - np.square((np.tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])))) * x[0]
    gradientVector[12] = w[10] * (1 - np.square((np.tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])))) * x[1]
    gradientVector[13] = w[10] * (1 - np.square((np.tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])))) * x[2]
    gradientVector[14] = w[10] * (1 - np.square((np.tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14]))))

    gradientVector[15] = 1 #weight is outside of all tanh's, and has coefficient of 1

    return gradientVector #output is flat 16 x 1 vector

def NonLinearFunction1(x):  #nonlinear function specified in assignment
    output = x[0] * x[1] + x[2]
    return output

def NonLinearFunction2(x): #my own function that is linear
    output = x[0] + x[1] + x[2]
    return output


def VectorNorm(x): #get norm of vector by squaring each element, then finding the root of their sum
    sum = 0
    for i in range(x.shape[0]):
        sum = sum + np.square(x[i])
    norm = np.sqrt(sum)
    return norm

def LossFunctionResidualPass(x, w, lmbda, nonLinearFunction): #find the loss vector resulting from inputs x (training data points), w (weights), lmbda (reg. loss constant), and the nonlinear function that is applied to the data points

    numberOfPoints = x.shape[0] #get number of points from height of matrix
    numberOfWeights = w.shape[0] #get number of weights

    residuals1 = np.zeros(numberOfPoints) #error between nonlinear map on training data points and neural network predicted map
    residuals2 = np.zeros(numberOfWeights) #error of regularization term on weights, essentially how "large" the norm is of the weights vector
    
    for row in range(numberOfPoints):
        residual = NeuralNetworkPass(x[row], w) - nonLinearFunction(x[row]) #find difference between what the network outputs and what it should have output
        residuals1[row] = residual #add to the residual vector

    for row in range(numberOfWeights):
        residual = np.sqrt(lmbda) * w[row]
        residuals2[row] = residual

    residuals = np.concatenate((residuals1, residuals2)) #construct the whole residual vector

    return residuals

def CalculateLoss(x, w, lmbda, nonLinearFunction): #x is N x 3 a collection of randomly generated points from non linear function; w is 16 x 1; lmbda (lambda) is constant
    residuals = LossFunctionResidualPass(x, w, lmbda, nonLinearFunction) #find the residual vector, essentially the diff. between what the neural network output and what it ideally should have
    residualsNormSquared = np.square(VectorNorm(residuals)) #find the norm of the residual vector, essentially how "big" the error is. squaring the norm gives squared error criteria
    return residualsNormSquared

def CalculateLossJacobian(x, w, lmbda): #x is N x 3 a collection of randomly generated points from non linear function; w is 16 x 1
    numberOfPoints = x.shape[0] #get number of points from height of matrix, N
    numberOfWeights = w.shape[0] #get number of weights
    outputJacobian  = np.zeros((numberOfPoints, numberOfWeights)) #N x 16 matrix

    for row in range(numberOfPoints):
        outputJacobian[row] = CalculateNeuralNetworkGradient(x[row], w) #the first N rows of the Jacobian are the transpose of the gradient vectors

    lambdaDiagonal = np.zeros((numberOfWeights, numberOfWeights))

    for i in range(numberOfWeights):
        lambdaDiagonal[i][i] = lmbda #construct the diagonal reg. loss lambda part of the Jacobian; the last 16 (because of 16 weights) rows of the Jacobian are the transpose of the regularization term's gradient vectors

    outputJacobian = np.vstack((outputJacobian, lambdaDiagonal)) #add the diagonal to the bottom of the Jacobian

    return outputJacobian

def pinv(A): #find pseudo-inverse of input matrix A
    U, s, V_transpose = scipy.linalg.svd(A) #use scipy SVD function to decompose A; s is a 1D array of singular values, NOT sigma

    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)

    m = A.shape[0]
    n = A.shape[1]

    sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        sigma[i, i] = s[i] #reconstruct sigma matrix using given singular values

    sigma_inverse = np.zeros((n,m)) #inverse of sigma is the reciprical of its elements; it is "trying its best" to get an identity matrix when multiplied with sigma
    for i in range(min(m, n)):
        if sigma[i, i] > 0: #check for non zero to avoid divide by zero error
            sigma_inverse[i, i] = 1 / sigma[i,i]

    A_pinv = np.matmul(V, sigma_inverse)
    A_pinv = np.matmul(A_pinv, U_transpose) #pseudo inverse of A is the inverse of its SVD, which is V * Sigma^-1 * U^T

    return A_pinv

def SolveNormalEquation(A, b): #min ||Ax - b||

    #closed form of normal equation solution is x = (A_T * A)^-1 * A_T * b

    ATA = np.matmul(np.transpose(A), A)
    ATA_inv = pinv(ATA) #pseudo inverse of A_T * A is the actual inverse since the matrix will always be non-singular (square and full rank)
    x = np.matmul(np.transpose(A), b)
    x = np.matmul(ATA_inv, x)
    return x

def EvaluatePerformance(inputPoints, weights, nonLinearFunction):

    #first see what the neural network predicts with input weights, and a test point
    #second compare output to g(test point), where g(x) is the non linear map that the network is trying to best approximate
    #the difference between the network output and g(test point) is the error
    #we square the error
    #then do this for all test points; accumulate the squared errors for a final squared error

    sumOfErrorSquared = 0

    for i in range(inputPoints.shape[0]):
        groundTruth = nonLinearFunction(inputPoints[i]) #what the network should be outputting, g(test point)
        prediction = NeuralNetworkPass(inputPoints[i], weights) #what the network outputs
        errorSquared = np.square(groundTruth - prediction)
        sumOfErrorSquared = sumOfErrorSquared + errorSquared
    
    return sumOfErrorSquared

def TrainNetwork(trainingData, iterations, initialWeights, lossLambda, initialTrustLambda, stopLossThreshold = 0.1, enableAutoStop = False):

    #train network iteratively using Levenberg-Marquardt algorithm

    print("Training network!\n")

    trustLambda = np.copy(initialTrustLambda) #pass in trust lambda by value because it will be updated per training iteration
    #trust lambda in LM algorithm regulates how far the weights "jump" for every iteration; this is determined by if the loss function is actually going down. If it's going down, then do smaller jumps in the weights to avoid it "missing" a local minima

    currentLoss = [] #keeps track of the iterative loss for plotting purposes
    w_k = initialWeights #keeps track of the weights that yield the lowest loss

    for iteration in range(iterations):

        k_loss = CalculateLoss(trainingData, w_k, lossLambda, NonLinearFunction1) #loss before approximation minimization

        lossJacobian = CalculateLossJacobian(trainingData, w_k, lossLambda) #get the Jacobian at current weights vector point for first order Taylor approximation

        #refer to page 391 of textbook

        #now the problem reduces to ordinary linear least squares. We are tring to minimize the norm squared of the 1st order Taylor approximation of the loss function, with a lambda trust regularization term


        A_identity = trustLambda * np.identity(lossJacobian.shape[1]) # (trust lambda)^0.5 * I for diagonal of trust lambda
        A = np.vstack((lossJacobian, A_identity)) #stack Df(x) ontop of (trust lambda)^0.5 * I, x represents weights, f is loss function

        #A matrix in normal equation can be constructed using the trust lambda and the Jacobian

        b_top = np.matmul(lossJacobian, w_k) - LossFunctionResidualPass(trainingData, w_k, lossLambda, NonLinearFunction1) # Df(x) * x - f(x); x is weights, f(x) is loss function
        b_bottom = np.sqrt(trustLambda) * w_k # (trust lambda) ^ 0.5 * x, x is weights
        b = np.concatenate((b_top, b_bottom)) #stack column vectors on top of each other is concat in numpy

        #b vector in normal equation can be constructed using the Jacobian, the loss residual vector, input training data points, and the trust lambda

        w_kplus1 = SolveNormalEquation(A, b) #solve normal equation to find the next weights vector; this vector minimizes the 1st order approx. of the loss function

        kplus1_loss = CalculateLoss(trainingData, w_kplus1, lossLambda, NonLinearFunction1) #loss after approximation minimization

        currentLoss.append(kplus1_loss) #add k+1 loss tracking array for plotting

        print(kplus1_loss)

        if enableAutoStop == True and currentLoss <= stopLossThreshold: #if current loss is below threshold, stop training
            break 

        #LM algorithm specifies how to determine the next iteration's trust lambda and weights

        if kplus1_loss <= k_loss: #loss function is actually going down
            trustLambda = 0.9 * trustLambda #decrease trust lambda so weights take smaller jumps
            w_k = w_kplus1 #set the next iteration's weights as this iteration's minimizing weights
        else: #loss function went up
            trustLambda = 1.1 * trustLambda #keep the same weights vector point, but now take a bigger jump, hoping that the actual loss will go down next iteration

        #w_k will always be the "best" set of weights that minimizes the loss function

    print("Done training network!\n")

    return w_k, currentLoss

def GenerateRandomPoints(numberOfPoints, bound): #utility for generating random points; bound will determine how "wide" the random spread of the points are

    RandomPoints = np.zeros((numberOfPoints, 3))

    for point in range(RandomPoints.shape[0]):
        RandomPoints[point] = np.random.uniform(-bound, bound, 3) #each point's 3 elements are sampled from a uniform dist., within specified bound

    return RandomPoints

