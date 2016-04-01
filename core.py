import neuralNetwork as nn
import random

# Preprocess the input file of mushrooms
def getLabeledData():
    mushroomFile = open("data.txt", "r")
    fileLines = mushroomFile.read().splitlines()
    unconvertedData = [[mushroom.strip() for mushroom in line.split(',')] for line in fileLines]

    # Convert alpha characters to numeric form
    numericMushroomFeatureVectors = [(alphaFeatureVector[0], [ord(feature)-ord('a') for feature in alphaFeatureVector[1:]]) for alphaFeatureVector in unconvertedData]
    return numericMushroomFeatureVectors


# Returns the percent correctly labeled by the classifier of the test set
def getAccuracy(classifier, mushrooms):
    correctlyClassifiedCount = sum([1 for label, mushroom in mushrooms if label == classifier(mushroom)])
    totalMushroomCount = len(mushrooms)

    print "Classified %i mushrooms correctly out of %i mushrooms possible" % (correctlyClassifiedCount, totalMushroomCount)
    return float(correctlyClassifiedCount) / totalMushroomCount


def getFoldAccuracy(labeledData, foldCount, mutationRate, crossoverRate, i):
        chunkSize = len(labeledData) / foldCount
        trainData = labeledData[0:i*chunkSize] + labeledData[chunkSize * (i+1):len(labeledData)]
        testData = labeledData[i*chunkSize:chunkSize*(i+1)]

        print "Mutation rate is %.0f%%. Crossover rate is %.0f%%" % (mutationRate*100, crossoverRate*100)
        classifier = nn.getClassifier(trainData, mutationRate, crossoverRate)
        return getAccuracy(classifier, testData)


# Uses N fold validation by training and testing on all possible permutations of the N data chunks
# Retrives statistics about accuracy of classifier for each attempt
def NFoldValidation(labeledData, foldCount):
    mutationRate = .05
    crossoverRate = .6

    summedAccuracy = sum([getFoldAccuracy(labeledData, foldCount, mutationRate, crossoverRate, i) for i in range(foldCount)])

    averageAccuracy = ((summedAccuracy / foldCount) * 100)
    print "Average accuracy for all validation is %.2f%%" % averageAccuracy


def main():
    labeledData = getLabeledData()
    random.shuffle(labeledData)
    NFoldValidation(labeledData, 2)

main()
