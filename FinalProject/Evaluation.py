import cv2
import numpy as np
from sklearn.metrics import mean_squared_error



class Evaluate:

    def __init__(self, groundTruth, predictions, focusedClusterValue, comparisonClusterValue):
        self.groundTruth = groundTruth
        self.predictions = predictions
        self.focusedVal = focusedClusterValue
        self.refVal = comparisonClusterValue
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def predictionError(self):
        err = mean_squared_error(self.groundTruth, self.predictions)
        
        print(f"Prediction error: {err}")
        print("______________________________________________")
        return err

    def sumAllDiff(self, groundTruth, predictions):
        # Find sum of all differences in all pairs in a two sorted arrays of index-corresponding numbers
        summ = 0
        if len(groundTruth) != len(predictions):
            raise ValueError('Ground truth-array and Prediction-array of inequal length! Returning sum=0')
        else:
            for index in range(0, len(predictions)):
                summ += abs(groundTruth[index] - predictions[index])

        print(f"Sum of all differences: {summ}")
        print("______________________________________________")
        return summ



    def confusionMatrix(self):
        # Predict when we see a portrait
        self.TP = 0  # True positive is when ground truth is a portrait and prediction is a portrait
        self.TN = 0  # True negative is when ground truth is a landscape and prediction is a landscape
        self.FP = 0  # False positive is when ground truth is landscape and prediction is portrait
        self.FN = 0  # False negative is when ground truth is portrait and prediction is landscape
        for index in range(len(self.groundTruth)):
            if self.groundTruth[index] == self.focusedVal and self.predictions[index] == self.focusedVal:
                self.TP += 1

            if self.groundTruth[index] == self.refVal and self.predictions[index] == self.focusedVal:
                self.FP += 1

            if self.groundTruth[index] == self.refVal and self.predictions[index] == self.refVal:
                self.TN += 1

            if self.groundTruth[index] == self.focusedVal and self.predictions[index] == self.refVal:
                self.FN += 1
        accuracy = ((self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)) * 100
        print("====================================================")
        print("Confusion matrix:")
        print("----------------------------------------------------")
        print(f"True Positives: {self.TP} || False Positives: {self.FP}")
        print("______________________________________________")
        print(f"False Negatives: {self.FN} || True Negatives: {self.TN}")
        print("----------------------------------------------------")
        print(f"Accuracy: {accuracy}")
        print("====================================================")
        return accuracy

    def precisionAndRecall(self):
        precision = self.TP / (self.TP + self.FP)
        recall = self.TP / (self.TP + self.FN)
        print("______________________________________________")
        print(f"Precision: {precision} || Recall: {recall}")
        print("______________________________________________")

        return precision, recall


if __name__ == "__main__":
    groundTruth = np.zeros([20])
    groundTruth[9:19] = 1
    print(groundTruth)
    print(len(groundTruth))
    predictions = [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    print(len(predictions))

    eval = Evaluate(groundTruth, predictions, 1, 0)
    eval.confusionMatrix()
    eval.precisionAndRecall()
