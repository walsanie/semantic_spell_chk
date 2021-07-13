"""
Created in 2016

@authors: Atheer Alkhalifa, Lamia Alkwai, Waleed Alsanie and Mohamed Alkanhal
         The National Center for Computation Technology & Applied Mathematics
         {aalkhalifa,lalkwai,walsanie, alkanhal} [at] kacst [dot] edu [dot] sa

This class is part of a context sensitive spell checking module. It is aimed to provide methods necessary
to interact with CRF++.
"""


from .any_learner import BaseLearner
import os
import sys
import re


class CRFPlusPlusInterface(BaseLearner):

    def __init__(self, error_marker):

        # check if CRF++ is installed
        x = os.popen(r"crf_learn")
        if not re.search("Yet Another", x.readline()):
            print("ERROR: CRF++ was not been detected in your system! Exiting ...")
            x.close()
            sys.exit(1)
        #

        x.close()
        super().__init__()
        self.__training_file = None
        self.__template = None
        self.__model_file = None
        self.__result_file = None
        # This list contains the correct label and the assigned label by the CRF++ as follows:
        # [(correct, assigned), .... ]
        self.__list_of_labels = list()
        # This is a list whose elements are contents from file from CRF++ test with probabilities option switched on.
        # Each element is a 4-tuple whose elements are the line number in the resulting file, correct label,
        # assigned label and the probability of the assigned label as follows
        # [ (line_n, correct, assigned, prob_assigned), .... ]
        self.__list_of_labels_with_probabilities = list()
        # This is the label identifier which is assigned to the spelling errors, e.g. 1. It is converted to str
        # because labels are read from a text file
        self.__error_identifier = str(error_marker)
        # This is the field number of the correct label in the file resulting from CRF++ in the test phase.
        self.__correct_label_field_no = 0
        # This will be true only if the testing is done with printing probabilities is chosen.
        self.__with_probabilities_mode = False

    def __compute_correct_incorrect_detections(self):
        """
        This method computes the correct and incorrect detections of spelling errors.
        :return:
        """

        # If probability mode is switched on use self.__list_of_labels_with_probabilities
        if self.__with_probabilities_mode:
            for (_, orig_label, assigned_label, _) in self.__list_of_labels_with_probabilities:
                if assigned_label == self.__error_identifier:
                    if orig_label == self.__error_identifier:
                        self._correct_detections += 1
                    else:
                        self._incorrect_detections += 1
        # If probability mode is switched off use self.__list_of_labels_with_probabilities
        else:
            for (orig_label, assigned_label) in self.__list_of_labels:
                if assigned_label == self.__error_identifier:
                    if orig_label == self.__error_identifier:
                        self._correct_detections += 1
                    else:
                        self._incorrect_detections += 1

    def __count_number_of_spelling_errors(self):
        """
        This is a private method which counts the number of spelling errors in the test set. This method should be
        called before computing the recall.
        :return:
        """

        f = open(self.__result_file, 'rt', encoding="utf-8")
        for line in f:
            if not line.isspace():
                l = line.split()
                try:
                    if l[self.__correct_label_field_no] == self.__error_identifier:
                        self._total_errors += 1
                except IndexError:
                    continue

        f.close()

    def get_number_of_errors(self):
        """
        This method returns the number of spelling errors set for the test set.
        :return: The number of spelling errors in the test set.
        """

        return self._total_errors

    def __fill_list_of_labels(self, output_file):
        """
        This is a private method which fills an internal list with the correct labels and the assigned labels for
        each token in the resulting file from the CRF++ test.
        :param output_file: The name of the output file
        :return:
        """

        self.__list_of_labels = list()                                  # Refresh list
        f = open(output_file, "rt", encoding="utf-8")
        for line in f:
            if re.match("#.*", line) or line.isspace():                # if the line is a comment of empty
                continue
            else:
                line_list = line.split()
                self.__list_of_labels.append((line_list[self.__correct_label_field_no],
                                              line_list[self.__correct_label_field_no + 1]))

        f.close()

    def __fill_list_of_labels_with_probabilities(self, result_file):
        """
        This is a private method which fills an internal list with the correct labels and the assigned labels for
        each token in the test file. It also fills the probability of the assigned label and the line number in the
        file resulting from CRF++ for each token. This is called when self.__with_probabilities_mode is switched on
        :param result_file: The name of the file which has resulted from crf_test.
        :return:
        """

        self.__list_of_labels_with_probabilities = list()                   # Refresh list
        f = open(result_file, "rt", encoding="utf-8")
        line_number = 1
        for line in f:
            if re.match("#.*", line) or line.isspace():                    # if the line is a comment of empty
                line_number += 1
                continue
            else:
                line_list = line.split()
                [assigned, prob] = line_list[self.__correct_label_field_no + 1].split("/")
                prob = float(prob)
                self.__list_of_labels_with_probabilities.append((line_number, line_list[self.__correct_label_field_no],
                                                                 assigned, prob))
                line_number += 1

        f.close()

    def call_objective_for_finding_c(self, objective="F-measure", beta=1):
        """
        This method calls the objective function for 'C' parameter in the CRF++ learning when 'C' is chosen to
        be estimated. Accepted values are 'F-measure' (the default value), 'Precision' and 'Recall'
        :param objective: Default is 'F-measure'
        :param beta: This is the beta parameter for computing the F-measure. It will only be considered if
        objective='F-measure'.
        :return: The value of the objective function.
        """

        recall = self.compute_recall()
        if re.match("[rR][eE][cC][aA][lL][lL]$", objective):
            return recall

        precision = self.compute_precision()
        if re.match("[pP][rR][eE][cC][iI][sS][iI][oO][nN]$", objective):
            return precision

        return self.compute_f_measure(precision, recall, beta)

    def train(self, template, training, model, a='CRF-L2', c=1, f=1):
        """
        This method trains CRF++ with the template file 'template and the training file 'training'. The generated
        model file will be 'model'. The training will be done with respect to the parameters specified by a, c and f.
        These parameters are explained in CRF++ manual in the url:
        https://taku910.github.io/crfpp/
        :param template: Template file
        :param training: Training file
        :param model: Model file
        :param a:Default is 'CRF-L2'
        :param c: Default is 1.
        :param f:Default is 1
        :return: None
        """

        self.__training_file = training
        self.__template = template
        self.__model_file = model
        if re.match("[cC][rR][fF][-][Ll]1$", a):
            a = 'CRF-L1'
            print("Training with a = 'CRF-L1, c = " + str(c) + " and f = " + str(f))
            os.system(r"crf_learn -a " + a + " -c " + str(c) + " -f " + str(f) + " " + template + " " + training +
                      " " + model)
        elif re.match("[cC][rR][fF][-][Ll]2$", a):
            print("Training with a = 'CRF-L2, c = " + str(c) + " and f = " + str(f))
            os.system(r"crf_learn -a " + a + " -c " + str(c) + " -f " + str(f) + " " + template + " " + training +
                      " " + model)
        else:
            a = 'CRF-L2'
            print("Unknown option for a! Training with the default option")
            print("Training with a = 'CRF-L2, c = " + str(c) + " and f = " + str(f))
            os.system(r"crf_learn -a " + a + " -c " + str(c) + " -f " + str(f) + " " + template + " " + training +
                      " " + model)

    def test(self, test_file, result_file, model_file=None, probabilities=False):
        """
        This method performs the testing on 'testing_file'. If 'model_file' is given it uses it for the testing,
        otherwise, it will use any model file set previous by the method 'CRFPlusPlusInterface.train'. The
        testing can be performed such that the resulting output prints the probabilities of assigned labels. This
        can be achieved by switching on the parameter 'probabilities'.
        :param test_file: Test file.
        :param result_file: The file to which the label of the test set will be written.
        :param model_file: Model file. Default is None, which means it will use the model generated from the training
        phase.
        :param probabilities: Test with probabilities of assigned label printing switched on.
        :return: None
        """

        # check if test file exists
        if not os.path.isfile(test_file):
            print("ERROR: Test file does not exist! Exiting the system.")
            sys.exit(1)

        self.__result_file = result_file
        # To identify the field number of the correct label
        t_file = open(test_file, 'rt', encoding="utf-8")
        self.__correct_label_field_no = len(t_file.readline().split()) - 1
        t_file.close()
        #

        if model_file:                                                  # if model file has been passed
            self.__model_file = model_file
        # Id not model file passed and it has not set before, perhaps through train.
        elif not self.__model_file:
            print("ERROR: Cannot test! Model file has not been specified. You either need to pass it, or train "
                  "a model to generate it.")
            return

        self.__with_probabilities_mode = probabilities
        if self.__with_probabilities_mode:
            os.system(r"crf_test -v1 -m " + self.__model_file + " " + test_file + " > " + result_file)
            self.__fill_list_of_labels_with_probabilities(result_file)
            print("Finished testing with probability mode switched on.")
        else:
            os.system(r"crf_test -m " + self.__model_file + " " + test_file + " > " + result_file)
            self.__fill_list_of_labels(result_file)
            print("Finished testing")

        # Set the number of detections and the number of total errors.
        self.__compute_correct_incorrect_detections()
        self.__count_number_of_spelling_errors()

    def write_lines_for_tokens_with_assignment_less_than(self, certainty_less_than, not_certain_file):
        """
        This method prints the line number, the token and the correct label, the assigned label and the probability
        of the assigned label of those assignment whose probabilities are less than the value given by
        'certainty_less_than'
        :param certainty_less_than: This specifies the upper bound of the probabilities of the assigned labels
        which will be printed out. This value must be between 0.5 and 1.0
        :param not_certain_file: The file to which to print.
        :return:
        """

        # Assigned probability cannot be less than 0.5 and greater than 1.
        if 1 < certainty_less_than < 0.5:
            print("ERROR: The certainty level value is not acceptable! It has to be some value in [0.5 - 1.0].")
            return

        f_out = open(not_certain_file, 'wt', encoding="utf-8")

        f_out.write('{0:20}{1:<20}{2:<20}{3:<30}'.format('LINE', 'CORRECT LABEL', 'ASSIGNED LABEL',
                                                         'PROBABILITY OF ASSIGNED LABEL') + '\n')
        f_out.write('{0:_<95}'.format('') + '\n')
        for (line, correct, assigned, prob) in self.__list_of_labels_with_probabilities:
            if float(prob) < certainty_less_than:
                f_out.write('{0:20}{1:<20}{2:<20}{3:<30}'.format(str(line), correct, assigned, str(prob)) + '\n')

        f_out.close()

    def compute_results(self, beta=1):
        """
        This method prints out the test results in the standard output.
        :param beta: The Beta value used to compute the f-measure. Default is 1.
        :return:
        """

        if not self.__result_file:
            print("ERROR: No result file has been generated from the testing! Please make sure that the CRF++ "
                  "test has been performed correctly")
            sys.exit(1)

        precision = self.compute_precision()
        recall = self.compute_recall()
        results = open("final_result.txt", 'a+', encoding="utf-8")

        results.write("\n" +'{0:#<41}'.format(''))
        results.write("\n" +'{0:20}{1:<20}{2}'.format('Precision:', str(100 * precision), "#"))
        results.write("\n" +'{0:20}{1:<20}{2}'.format('Recall:', str(100 * recall), "#"))
        results.write("\n" +'{0:20}{1:<20}{2}'.format('F-measure:', str(100 * self.compute_f_measure(precision, recall, beta)), "#"))
        results.write("\n" +'{0:#<41}'.format(''))








