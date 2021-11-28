from os import error
import scipy.linalg
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


def NeuralNetworkPass(x, w): #x is 3 x 1, w is 16 x 1 vectors
    
    perceptronOneOutput = w[0] * np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])
    perceptronTwoOutput = w[5] * np.tanh(w[6]*x[0] + w[7]*x[1] + w[8]*x[2] + w[9])
    perceptronThreeOutput = w[10] * np.tanh(w[11]*x[0] + w[12]*x[1] + w[13]*x[2] + w[14])

    output = perceptronOneOutput + perceptronTwoOutput + perceptronThreeOutput + w[15]

    return output

def CalculateNeuralNetworkGradient(x, w): #x is 3 x 1, w is 16 x 1 vectors
    gradientVector = np.zeros(16)

    gradientVector[0] = np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])
    gradientVector[1] = w[0] * (1 - np.square((np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])))) * x[0]
    gradientVector[2] = w[0] * (1 - np.square((np.tanh(w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4])))) * x[1]
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

    gradientVector[15] = 1

    #print(gradientVector)

    return gradientVector #output is flat 16 x 1 vector

def NonLinearFunction1(x):
    output = x[0] + x[1] + x[2]
    return output

def VectorNorm(x):
    sum = 0
    for i in range(x.shape[0]):
        sum = sum + np.square(x[i])
    norm = np.sqrt(sum)
    return norm


def LossFunctionResidualPass(x, w, lmbda, nonLinearFunction):
    numberOfPoints = x.shape[0] #get number of points from height of matrix
    numberOfWeights = w.shape[0] #get number of weights

    residuals1 = np.zeros(numberOfPoints)
    residuals2 = np.zeros(numberOfWeights)
    
    for row in range(numberOfPoints):
        residual = NeuralNetworkPass(x[row], w) - nonLinearFunction(x[row])
        residuals1[row] = residual

    for row in range(numberOfWeights):
        residual = np.sqrt(lmbda) * w[row]
        residuals2[row] = residual

    residuals = np.concatenate((residuals1, residuals2))

    return residuals

def CalculateLoss(x, w, lmbda, nonLinearFunction): #x is N x 3 a collection of randomly generated points from non linear function; w is 16 x 1; lmbda (lambda) is constant
    residuals = LossFunctionResidualPass(x, w, lmbda, nonLinearFunction)
    residualsNormSquared = np.square(VectorNorm(residuals))
    return residualsNormSquared

def CalculateLossJacobian(x, w, lmbda): #x is N x 3 a collection of randomly generated points from non linear function; w is 16 x 1
    numberOfPoints = x.shape[0] #get number of points from height of matrix
    numberOfWeights = w.shape[0] #get number of weights
    outputJacobian  = np.zeros((numberOfPoints, numberOfWeights)) #N x 16 matrix

    for row in range(numberOfPoints):
        outputJacobian[row] = CalculateNeuralNetworkGradient(x[row], w)


    #print(outputJacobian.shape)

    lambdaDiagonal = np.zeros((numberOfWeights, numberOfWeights))

    for i in range(numberOfWeights):
        lambdaDiagonal[i][i] = lmbda

    outputJacobian = np.vstack((outputJacobian, lambdaDiagonal))

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
    ATA = np.matmul(np.transpose(A), A)
    ATA_inv = pinv(ATA)
    x = np.matmul(np.transpose(A), b)
    x = np.matmul(ATA_inv, x)
    return x


def EvaluatePerformance(inputPoints, weights, nonLinearFunction):

    sumOfErrorSquared = 0

    for i in range(inputPoints.shape[0]):
        groundTruth = nonLinearFunction(inputPoints[i]) #what the network should be outputting
        prediction = NeuralNetworkPass(inputPoints[i], weights)
        errorSquared = np.square(groundTruth - prediction)
        sumOfErrorSquared = sumOfErrorSquared + errorSquared
    
    return sumOfErrorSquared

        

    




x = np.random.rand(3)


xRandomPoints = np.zeros((500, 3))

for point in range(xRandomPoints.shape[0]):
    xRandomPoints[point] = np.random.uniform(-1, 1, 3)

print(xRandomPoints)




#print(NeuralNetworkPass(x, w))
#print(CalculateNeuralNetworkGradient(x, w))
#print(CalculateLoss(xRandomPoints, w, 0.5, NonLinearFunction1))

iterationCount = 200



lossLambdas = [0.00001, 0.0001, 0.01, 0.1, 1, 10]

lossThreshold = 0.25

iterations = np.arange(iterationCount)



xRandomTestPoints = np.zeros((100, 3))

for point in range(xRandomTestPoints.shape[0]):
    xRandomTestPoints[point] = np.random.uniform(-1, 1, 3)



for lossLambda in lossLambdas:

    print("new")

    loss = []
    w_k = np.random.uniform(-10, 10, 16)
    trustLambda = 1

    for iteration in iterations:


        k_loss = CalculateLoss(xRandomPoints, w_k, lossLambda, NonLinearFunction1)

        #loss.append(k_loss)
        #print(k_loss)


        #print(CalculateLoss(xRandomPoints, w_k, lossLambda, NonLinearFunction1))
        #print(trustLambda)

        lossJacobian = CalculateLossJacobian(xRandomPoints, w_k, lossLambda)

        A_identity = trustLambda * np.identity(lossJacobian.shape[1])
        A = np.vstack((lossJacobian, A_identity))


        b_top = np.matmul(lossJacobian, w_k) - LossFunctionResidualPass(xRandomPoints, w_k, lossLambda, NonLinearFunction1)
        b_bottom = np.sqrt(trustLambda) * w_k
        b = np.concatenate((b_top, b_bottom)) #stack column vectors on top of each other is concat in numpy
        w_kplus1 = SolveNormalEquation(A, b)

        #if np.array_equal(w_kplus1, w_k):
            #break

        kplus1_loss = CalculateLoss(xRandomPoints, w_kplus1, lossLambda, NonLinearFunction1)

        print(kplus1_loss)
        loss.append(kplus1_loss)


        #if k_loss < lossThreshold:
         #   break
        #print(kplus1_loss)
        #print(kplus1_loss <= k_loss)

        if kplus1_loss <= k_loss:
            trustLambda = 0.9 * trustLambda
            w_k = w_kplus1
        else:
            trustLambda = 1.1 * trustLambda
            
    performance = EvaluatePerformance(xRandomTestPoints, w_k, NonLinearFunction1)
    plt.plot(iterations, loss, label = str(lossLambda) + "performance: " + str(performance))

plt.title("Loss vs iteration")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()


'''

randomRanges = [0.1, 100]


for randomRange in randomRanges:

    for i in range(3):

        print("new")

        loss = []
        w_k = np.random.uniform(-randomRange, randomRange, 16)
        trustLambda = 1
        lossLambda = 0.001

        for iteration in iterations:


            k_loss = CalculateLoss(xRandomPoints, w_k, lossLambda, NonLinearFunction1)

            #loss.append(k_loss)
            #print(k_loss)


            #print(CalculateLoss(xRandomPoints, w_k, lossLambda, NonLinearFunction1))
            #print(trustLambda)

            lossJacobian = CalculateLossJacobian(xRandomPoints, w_k, lossLambda)

            A_identity = trustLambda * np.identity(lossJacobian.shape[1])
            A = np.vstack((lossJacobian, A_identity))


            b_top = np.matmul(lossJacobian, w_k) - LossFunctionResidualPass(xRandomPoints, w_k, lossLambda, NonLinearFunction1)
            b_bottom = np.sqrt(trustLambda) * w_k
            b = np.concatenate((b_top, b_bottom)) #stack column vectors on top of each other is concat in numpy
            w_kplus1 = SolveNormalEquation(A, b)

            #if np.array_equal(w_kplus1, w_k):
                #break

            kplus1_loss = CalculateLoss(xRandomPoints, w_kplus1, lossLambda, NonLinearFunction1)

            print(kplus1_loss)
            loss.append(kplus1_loss)


            #if k_loss < lossThreshold:
            #   break
            #print(kplus1_loss)
            #print(kplus1_loss <= k_loss)

            if kplus1_loss <= k_loss:
                trustLambda = 0.9 * trustLambda
                w_k = w_kplus1
            else:
                trustLambda = 1.1 * trustLambda
        
        plt.plot(iterations, loss, label = str(randomRange) + str(i))

plt.title("Loss vs iteration")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

'''


'''


numberOfTestPoints = [50, 150, 300, 500, 1000]

performance = []

w_k = np.random.uniform(-10, 10, 16)
trustLambda = 1
lossLambda = 0.001

for iteration in range(300):


    k_loss = CalculateLoss(xRandomPoints, w_k, lossLambda, NonLinearFunction1)

    #loss.append(k_loss)
    #print(k_loss)


    #print(CalculateLoss(xRandomPoints, w_k, lossLambda, NonLinearFunction1))
    #print(trustLambda)

    lossJacobian = CalculateLossJacobian(xRandomPoints, w_k, lossLambda)

    A_identity = trustLambda * np.identity(lossJacobian.shape[1])
    A = np.vstack((lossJacobian, A_identity))


    b_top = np.matmul(lossJacobian, w_k) - LossFunctionResidualPass(xRandomPoints, w_k, lossLambda, NonLinearFunction1)
    b_bottom = np.sqrt(trustLambda) * w_k
    b = np.concatenate((b_top, b_bottom)) #stack column vectors on top of each other is concat in numpy
    w_kplus1 = SolveNormalEquation(A, b)

    #if np.array_equal(w_kplus1, w_k):
        #break

    kplus1_loss = CalculateLoss(xRandomPoints, w_kplus1, lossLambda, NonLinearFunction1)

    print(kplus1_loss)


    #if k_loss < lossThreshold:
    #   break
    #print(kplus1_loss)
    #print(kplus1_loss <= k_loss)

    if kplus1_loss <= k_loss:
        trustLambda = 0.9 * trustLambda
        w_k = w_kplus1
    else:
        trustLambda = 1.1 * trustLambda
'''

'''

for numberOfTestPoint in numberOfTestPoints:


    xRandomTestPoints = np.zeros((numberOfTestPoint, 3))

    for point in range(xRandomTestPoints.shape[0]):
        xRandomTestPoints[point] = np.random.uniform(-1, 1, 3)

    performance.append(EvaluatePerformance(xRandomTestPoints, w_k, NonLinearFunction1))



print("evaluation \n")

print(EvaluatePerformance(xRandomTestPoints, w_k, NonLinearFunction1))



plt.plot(numberOfTestPoints, performance)

plt.title("performance vs number of test points")
plt.xlabel('test points')
plt.ylabel('performance')
plt.show()

'''
















