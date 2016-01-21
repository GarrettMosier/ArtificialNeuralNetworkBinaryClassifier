import math, random

# Determines if a neuron fires
def activate(nodeInput, weights):
    assert len(nodeInput) == len(weights)
    return 1 if sum([x * y for x, y in zip(nodeInput, weights)]) > 0 else 0

# Mathematical sigmoid function. Used for activation and non-linearity
def sigmoid(n):
    return 1 / (1 + math.exp(-n))

# Create a three dimensional matrix with first index as layerNo
# Leaves of the matrix are weights for a given edge in the neural network
def getInitialWeights(nodesPerLayer):
    return [[[random.random() for j in range(nodesPerLayer[i])] for k in range(nodesPerLayer[i+1] if i != len(nodesPerLayer) - 1  else 1)] for i in range(len(nodesPerLayer))]

# Calculates how well a given classifier can predict the label on a data set
def getAccuracy(trainingData, classifier):
    return sum([1 for label, mushroom in trainingData if label == classifier(mushroom)])

# Ranks population weights based on how well the ANN performs
def getRankedPopulation(trainingData, populationWeights):
    classifiers = [makeClassifierWithWeights(weights) for weights in populationWeights]
    accuracies = [getAccuracy(trainingData, classifier) for classifier in classifiers]

    # Put best weights at front
    rankedPopulation = sorted(zip(accuracies, populationWeights), key=lambda t: t[0])
    rankedPopulation.reverse()

    return [weights for acc, weights in rankedPopulation]

# Uses the a combination of steady-state and elitism selection methods
def updateGeneticWeights(trainingData, populationWeights, populationSize, mutationRate, crossoverRate):
    rankedPopulation = getRankedPopulation(trainingData, populationWeights)
    populationWeights = getNextGeneration(rankedPopulation, mutationRate, crossoverRate)
    populationWeights = getRankedPopulation(trainingData, populationWeights)

    return populationWeights

# Creates a classifier with weights for each layer of the neural network
def makeClassifierWithWeights(layerWeights):
    def classifier(mushroom):
        layerOutput = list()
        layerOutput.append(mushroom)

        for layerNo in range(len(layerWeights)):
            layerOutput.append([activate(layerOutput[layerNo], rowWeight) for rowWeight in layerWeights[layerNo]])

        # Get the single value from the last layer of the output matrix
        assert(len(layerOutput[-1]) == 1)
        return 'p' if layerOutput[-1][0] else 'e'

    return classifier

def traverse(weightLevel, operate):
    newWeights = list()
    if type(weightLevel) == list:
        for i in range(len(weightLevel)):
            if type(weightLevel[i]) == list:
                newWeights.append(traverse(weightLevel[i], operate))
            else:
                newWeights.append(operate(weightLevel[i]))
    else:
        return newWeights.append(operate(weightLevel))

    return newWeights

#w1 and w2 must have the same structure
def zipWeights(w1, w2):
    newWeights = list()

    if type(w1) == list:
        for i in range(len(w1)):
            if type(w1[i]) == list:
                newWeights.append(zipWeights(w1[i], w2[i]))
            else:
                newWeights.append((w1[i], w2[i]))
    else:
        return newWeights.append((w1, w2))

    return newWeights

# Combine two entities on a weight by weight basis
def crossover(w1, w2, crossoverRate):
    return traverse(zipWeights(w1, w2), lambda x : x[0] if random.random() > crossoverRate else x[1])

# Pick a random operator and apply it to mutationRate percent of weights
def mutate(weights, mutationRate):
    operatorList = [lambda x, y: x + y, lambda x, y: x * y, lambda x, y: x / y, lambda x, y: x - y]
    return traverse(weights, lambda x: x if random.random() > mutationRate else operatorList[random.randrange(0, 4)](x, random.randrange(1, 100, 1)))

def getNextGeneration(rankedPopulation, mutationRate, crossoverRate):
    outlierSize = 1

    parents = rankedPopulation[:outlierSize]
    basicPopulation = rankedPopulation[outlierSize:-outlierSize]

    # Crossover. Combine every parent with every other suitable member of the population
    crossovers = [crossover(parents[i], basicPopulation[j], crossoverRate) for i in range(len(parents)) for j in range(len(basicPopulation))]

    nextGen = parents + [mutate(weights, mutationRate) for weights in crossovers + parents]

    return nextGen

# Uses a three layer neural network to create a classifier
def getClassifier(trainingData, mutationRate, crossoverRate):
    inputSize = len(trainingData[0][1])
    nodesPerLayer = [inputSize, 1]

    populationSize = 60
    print "Starting population size is %i" % populationSize

    populationWeights = [getInitialWeights(nodesPerLayer) for i in range(populationSize)]
    print "Initial weights for the entire population are ", populationWeights

    for i in range(100):
        populationWeights = updateGeneticWeights(trainingData, populationWeights, populationSize, mutationRate, crossoverRate)
        assert(len(populationWeights) == populationSize)

    print "Final population weights are ", populationWeights

    return makeClassifierWithWeights(populationWeights[0])
