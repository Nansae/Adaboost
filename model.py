import math
import numpy as np

class Adaboost(object):
    def __init__(self):
        self.weakClassifiers = []
        self.StrongClassifiers = []

    def train(self, data, label, T = 50):
        dim = data.shape[1]
        N = data.shape[0]
        D = np.array([1/N for w in range(N)], dtype=np.float)
        
        for d in range(dim):
            row = data[0:N, 0:d]
            sorted_row = np.sort(row)

            for i in range(N-1):
                self.weakClassifiers.append([d, (sorted_row[i] + sorted_row[i + 1]) / 2.0, 1])
                self.weakClassifiers.append([d, (sorted_row[i] + sorted_row[i + 1]) / 2.0, -1])

        print("Weak Classifier:", len(self.weakClassifiers))

        for t in range(T):
            minError = 9999.0
            minErrorWeakClassifier = None
            for w in range(len(self.weakClassifiers)):
                error = 0.0
                for i in range(N):
                    if self.weakClassifiers[w][1] > data[i, self.weakClassifiers[w][0]]:
                        if label[i] != self.weakClassifiers[w][2]:
                            error = error + D[i]
                    else:
                        if label[i] != (-1*self.weakClassifiers[w][2]):
                            error = error + D[i]
                if minError > error:
                    minError = error
                    minErrorWeakClassifier = self.weakClassifiers[w]

            minError += 0.000001
            alpha = 0.5 * math.log((1-minError)/minError)
            self.StrongClassifiers.append([alpha, minErrorWeakClassifier])

            for i in range(N):
                if minErrorWeakClassifier[1] > data[i, minErrorWeakClassifier[0]]:
                    if minErrorWeakClassifier[2] != label[i]:
                        D[i] = D[i] * math.exp(alpha)
                    else:
                        D[i] = D[i] * math.exp(-alpha)
                else:
                    if -1 * minErrorWeakClassifier[2] != label[i]:
                        D[i] = D[i] * math.exp(alpha)
                    else:
                        D[i] = D[i] * math.exp(-alpha)

            D/=sum(D)

            print("Error: %4f alpha: %4f" % (minError, alpha))

    def predict(self, input):
        sigma = 0
        for Classifier in self.StrongClassifiers:
            a = Classifier[0]
            if Classifier[1][1] > input[Classifier[1][0]]:
                h = Classifier[1][2]
            else:
                h = -Classifier[1][2]
            sigma += (a*h)

        if sigma < 0:
            return -1
        else:
            return 1