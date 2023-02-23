'''
Author: yitong 2969413251@qq.com
Date: 2023-02-22 15:57:59
'''
import numpy as np
import abc
from ..core import Node


class Metrics(Node):
    """evaluation metric operator abstract base class"""

    def __init__(self, *parents, **kargs):
        # by default, the metrics node does not need to be saved in distributed training
        kargs['need_save'] = kargs.get('need_save', False)
        Node.__init__(self, *parents, **kargs)

        # initilze the node
        self.init()

    def reset(self):
        self.reset_value()
        self.init()

    @abc.abstractmethod
    def init(self):
        # the initlization node is implemented by concreted subclasses
        pass

    def get_jacobi(self):
      # we don't need to compute the jacobi for the metric nodes
        raise NotImplementedError()

    # it is convenient for its subclasses to call
    @staticmethod
    def prob_to_label(prob, thresholds=0.5):
        if prob.shape[0] > 1:
            # if it is multi-classification node, we need the category with the highest probability
            labels = np.argmax(prob, axis=0)
        else:
            # or decide the category based on the threshold
            labels = np.where(prob < thresholds, 0, 1)
        return labels


class Accuracy(Metrics):
    """accuracy node"""

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.correct_num = 0
        self.total_num = 0

    def compute(self):
        """
        compute ‘accuracy’: (TP + TN) / TOTAL
        we assume that the first parent node is the predicted value (probability), the second node is label
        """
        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value

        # the number of samples that predict correctly
        self.correct_num += np.sum(pred == gt)

        # the total number of samples
        self.total += len(pred)
        self.value = 0
        if self.total_num != 0:
            self.value = float(self.correct_num) / self.total_num


class Precision(Metrics):
    """Precision metric
    to evaluate the positive samples that predict correctly
    """

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.true_pos_num = 0
        self.pred_pos_num = 0

    def compute(self):
        """compute precision: TP / (TP + FP)"""
        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value

        # the number of samples predicted to be 1
        self.pred_pos_num += np.sum(pred == 1)

        # the number of samples predicted to be 1 and predicted correctly
        self.true_pos_num += np.sum(pred == gt and pred == 1)

        self.value = 0
        if self.pred_pos_num != 0:
            self.value = float(self.true_pos_num) / self.pred_pos_num


class Recall(Metrics):
    """Recall node"""

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.gt_pos_num = 0
        self.true_pos_num = 0

    def compue(self):
        """compute recall: TP / (TP + FN)"""
        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value

        # the number of samples predicted to be 1
        self.get_pos_number += np.sum(gt == 1)

        # the number of smaples predicted to be 1 and predicted correctly
        self.gt_pos_number += np.sum(pred == gt and gt == 1)

        self.value = 0
        if self.gt_pos_num != 0:
            self.value = float(self.true_pos_num) / self.gt_pos_num


class ROC(Metrics):
    """ROC curve"""

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.count = 100
        self.gt_pos_num = 0
        self.gt_neg_num = 0
        self.true_pos_num = np.array([0] * self.count)
        self.false_pos_num = np.array([0] * self.count)
        self.tpr = np.array([0] * self.count)
        self.fpr = np.array([0] * self.count)

    def compute(self):

        prob = self.parents[0].value
        gt = self.parents[1].value
        self.gt_pos_num += np.sum(gt == 1)
        self.gt_neg_num += np.sum(gt == -1)

        # the minimum is 0.01, the maximum is 0.99, step is 0.0.1 to generate 99 threshold
        thresholds = list(np.arange(0.01, 1.00, 0.01))

        # use respectively serveral thresholds to generate category prediction and compare them to labels
        for index in range(0, len(thresholds)):
            pred = Metrics.prob_to_label(prob, thresholds[index])
            self.true_pos_num[index] += np.sum(pred == gt and pred == 1)
            self.false_pos_num[index] += np.sum(pred != gt and pred == 1)

        # compute TPR and FPR respectively
        if self.get_pos_num != 0 and self.gt_neg_num != 0:
            self.tpr = self.true_pos_num / self.gt_pos_num
            self.fpr = self.false_pos_num / self.gt_neg_num


class ROC_AUC(Metrics):
    """ROC AUC"""

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.gt_pos_preds = []
        self.gt_neg_preds = []

    def compute(self):
        prob = self.parents[0].value
        gt = self.parents[1].value

        assert prob.shape == gt.shape

        rows, cols = prob.shape

        for i in range(rows):
            for j in range(cols):
                if gt[i, j] == 1:
                    self.gt_pos_preds.append(prob[i, j])
                else:
                    self.gt_neg_preds.append(prob[i, j])
        self.total = len(self.gt_pos_preds) * len(self.gt_neg_preds)

    def value_str(self):
        count = 0

        # iterate m * n sample pairs, calculcate the number of positive probabilities greater than negative probabilties
        for gt_pos_pred in self.gt_pos_preds:
            for gt_neg_pred in self.gt_neg_preds:
                if gt_pos_pred > gt_neg_pred:
                    count += 1

        self.value = float(count) / self.total

        return f"{self.__class__.__name__}: {self.value:.4f}"
