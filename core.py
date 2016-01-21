import neuralNetwork as nn
import random

# Preprocess the input file of mushrooms
def getLabeledData():
    f = open("data.txt", "r")
    lines = f.read().splitlines()
    data = [[mushroom.strip() for mushroom in line.split(',')] for line in lines]
    return [(d[0], [ord(feature)-ord('a') for feature in d[1:]]) for d in data]

# Returns the percent correctly labeled by the classifier of the test set
def getAccuracy(classifier, mushrooms):
    correctlyClassified = sum([1 for label, mushroom in mushrooms if label == classifier(mushroom)])
    print "Classified %i mushrooms correctly out of %i mushrooms possible" % (correctlyClassified, len(mushrooms))
    return float(correctlyClassified) / len(mushrooms)

# Uses N fold validation by training and testing on all possible permutations of the N data chunks
# Retrives statistics about accuracy of classifier for each attempt
def NFoldValidation(labeledData, foldCount):
    mutationRate = .05
    crossoverRate = .6

    summedAccuracy = 0
    for i in range(foldCount):
        chunkSize = len(labeledData) / foldCount
        trainData = labeledData[0:i*chunkSize] + labeledData[chunkSize * (i+1):len(labeledData)]
        testData = labeledData[i*chunkSize:chunkSize*(i+1)]

        print "Mutation rate is %.0f%%. Crossover rate is %.0f%%" % (mutationRate*100, crossoverRate*100)
        classifier = nn.getClassifier(trainData, mutationRate, crossoverRate)
        summedAccuracy += getAccuracy(classifier, testData)

    print "Average accuracy for all validation is %.2f%%" % ((summedAccuracy / foldCount) * 100)

def main():
    labeledData = getLabeledData()
    random.shuffle(labeledData)
    NFoldValidation(labeledData, 10)

main()
