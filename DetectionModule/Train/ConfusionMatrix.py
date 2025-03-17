import numpy as np
import matplotlib.pyplot as plt


class ConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        """
	    normalize: Whether to set the element as a percentage
        """
        self.__normalize = normalize
        self.__labels_name = labels_name
        self.__num_classes = len(labels_name)
        self.__matrix = np.zeros((self.__num_classes, \
                        self.__num_classes), dtype="float32")


    def Update(self, labels, predicts):
        """
	    :param labels: actual label set, eg: array([0,5,0,6,2,...], dtype=int64)
        :param predicts: predictive label set, eg: array([0,5,1,6,3,...], dtype=int64)        
        :return:
        """
        for label, predict in zip(labels, predicts):
            self.__matrix[label, predict] += 1


    def GetMatrix(self, normalize=True):
        """
        returns: matrix 
        """
        if normalize:
            # calculate the sum of each row for percentage calculation
            per_sum = self.__matrix.sum(axis=1)
            for i in range(self.__num_classes):
                if per_sum[i] != 0:
                    self.__matrix[i] = (self.__matrix[i] / per_sum[i])   
            self.__matrix = np.around(self.__matrix, 2)

        return self.__matrix


    def Plot(self, cfm_path='./ConfusionMatrix.png'):
        """
        returns: plot the confusion matrix 
        """
        plt.clf()
        
        self.__matrix = self.GetMatrix(self.__normalize)

        plt.imshow(self.__matrix, cmap=plt.cm.Blues)  

        plt.yticks(range(self.__num_classes), self.__labels_name, fontsize=12)
        plt.xticks(range(self.__num_classes), self.__labels_name, fontsize=12, rotation=90)

        for x in range(self.__num_classes):
            for y in range(self.__num_classes):
                value = float(format('%.2f' % self.__matrix[x, y]))
                plt.text(y, x, value, verticalalignment='center', 
                                    fontsize=12, horizontalalignment='center')

        plt.colorbar()
        plt.savefig(cfm_path, bbox_inches='tight', dpi=200)
        plt.show()


    def UpdateAndPlot(self, labels, preds, cfm_path='./ConfusionMatrix.png'):
        """
        UpdateAndPlot: update the cfm and plot it
        :param labels: actual label set, eg: array([0,5,0,6,2,...], dtype=int64)
        :param predicts: predictive label set, eg: array([0,5,1,6,3,...], dtype=int64)     
        """
        self.Update(labels, preds)
        self.Plot(cfm_path)