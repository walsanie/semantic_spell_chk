"""
Created in 2016

@authors: Atheer Alkhalifa, Lamia Alkwai, Waleed Alsanie and Mohamed Alkanhal
         The National Center for Computation Technology & Applied Mathematics
         {aalkhalifa,lalkwai,walsanie, alkanhal} [at] kacst [dot] edu [dot] sa

This class is part of a context sensitive spell checking module. It is aimed to serve as a base class of the learning
algorithms employed to solve the problem.
"""


class BaseLearner:

    def __init__(self):
        self._correct_detections = 0                                        # protected member
        self._incorrect_detections = 0                                      # protected member
        self._total_errors = 0                                              # protected member

    def set_correct_detection(self, n):
        """
        This method sets the number of correctly detected errors.
        :param n: The number of correctly detected errors.
        :return: None
        """

        self._correct_detections = n

    def get_correct_detections(self):
        """
        Returns the current value of correct detection.
        :return: None
        """

        return self._correct_detections

    def set_incorrect_detection(self, n):
        """
        This method sets the number of incorrect detections.
        :param n: The number of incorrect detections.
        :return: None
        """

        self._incorrect_detections = n

    def get_incorrect_detections(self):
        """
        Returns the current value of incorrect detection.
        :return: None
        """

        return self._incorrect_detections

    def set_total_errors(self, n):
        """
        This method sets the total number of semantic seplling errors in the corpus.
        :param n: The total number of errors.
        :return: None
        """

        self._total_errors = n

    def get_total_errors(self):
        """
        Returns the total number of semantic errors in the corpus
        :return: Total number of semantic errors.
        """

        return self._total_errors

    def compute_precision(self):
        """
        This method computes the precision.
        :return: Precision
        """
        if self._correct_detections + self._incorrect_detections == 0:
            return 0
        else:
            return self._correct_detections/(self._correct_detections + self._incorrect_detections)

    def compute_recall(self):
        """
        This method computes the recall.
        :return: Recall.
        """

        return self._correct_detections/self._total_errors if self._total_errors else 0

    @staticmethod
    def compute_f_measure(precision, recall, beta=1):
        """
        This method is a static method which computes the F-measure value, given the values of the precision and recall
        with the specified value of Beta. The default value of Beta is 1.
        :param precision: The precision.
        :param recall: The recall
        :param beta: The value of Beta based on which the F-measure is computed. Default = 1.
        :return: F-measure.
        """

        if (precision + recall) == 0:
            return 0
        else:
            return (1 + beta ** 2) * precision * recall/(((beta ** 2) * precision) + recall)

