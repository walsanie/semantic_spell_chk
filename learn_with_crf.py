
import re, fnmatch
import sys
import math
import random
import os, shutil
from context_sensitive_spell_chk.preprocessing import Preprocessor
from context_sensitive_spell_chk.crf_pp_interface import CRFPlusPlusInterface
from subprocess import *
from nltk.stem.isri import ISRIStemmer


def format_crf_pp_file_semantic_features(file_name, preprocessing_obj, error_marker, correct_marker):

    f_out = open(file_name, 'wt', encoding="utf-8")
    sentences = preprocessing_obj.get_sentences()
    errors = preprocessing_obj.get_errors()
    t = 0
    try:

        for (sent_num, (sent, terminator)) in enumerate(sentences):
            words = preprocessing_obj.divide_sentence_into_words(sent)
            for (pos, wrd) in enumerate(words):
                st = ISRIStemmer()
                try:
                    s = check_output(["java", "-jar", "queryAWOntology.jar", wrd, st.stem(wrd), "", "0"])
                    s = s.decode(encoding="UTF-8")
                    s = s.split(",")[0].split("\n")[0].split(" ")
                    t = t+1
                except Exception:
                    s = "Some_Name_Or_Act"

                # For the words that didn't return  category
                if s[len(s)-1] == "":
                    s = "Some_Name_Or_Act"


                if (sent_num, pos) in errors:
                    f_out.write(wrd + "\t" + s[len(s)-1] + "\t" + error_marker + "\n")
                    print(t)

                else:
                    f_out.write(wrd + "\t" + s[len(s)-1]+ "\t" + correct_marker + "\n")
                    #print(wrd + "\t" + s[len(s)-1]+ "\t" + correct_marker + "\n")
                    print(t)

            f_out.write(terminator +"\t" + "###############################"+ "\t" + correct_marker + "\n\n")
    except ValueError:
            print("passed")
            pass
    f_out.close()


def format_crf_pp_file_pos_tags(file_name, preprocessing_obj, error_marker, correct_marker):

    file = open("temp.txt", 'w', encoding="utf-8")
    for s in preprocessing_obj.get_sentences():
        print(s[0]  + '/n'  , file=file)
    temp = check_output('java    -mx1g   -cp   stanford-postagger.jar:context_sensitive_spell_chk/lib/* edu.stanford.nlp.tagger.maxent.MaxentTagger    -model    context_sensitive_spell_chk/lib/arabic.tagger    -textFile temp.txt' , shell=True, stderr=PIPE)
    tagged_subset = temp.decode("utf-8").split("/n")
    #tagged_subset = preprocessing_obj.chop_off_text_into_sentences(temp.decode("utf-8"))

    file.close()


    f_out = open(file_name, 'wt', encoding="utf-8")
    sentences = preprocessing_obj.get_sentences()
    errors = preprocessing_obj.get_errors()
    st = ISRIStemmer()

    for (sent_num, (sent, terminator)), t_sent in zip(enumerate(sentences), tagged_subset ):

        words = preprocessing_obj.divide_sentence_into_words(sent)
        tagged_sent = preprocessing_obj.divide_sentence_into_words(t_sent)

        for (pos, wrd) in enumerate(words):
            try:
                tag = tagged_sent[tagged_sent.index(wrd)+2]
            except IndexError:
                tag = tagged_subset[sent_num+1].split("/")[1].split("\n")[0]
            except ValueError:
                continue
            if (sent_num, pos) in errors:
                f_out.write(wrd + "\t"  + st.stem(wrd)+  "\t" + tag +  "\t" + error_marker + "\n")
               # print(wrd + "\t" + tag + "\t" + error_marker + "\n")
            else:
                f_out.write(wrd + "\t" + st.stem(wrd)+  "\t" + tag +  "\t" + correct_marker + "\n")
              #  print(wrd + "\t" + tag + "\t" + correct_marker + "\n")

        f_out.write(terminator + "\t" + terminator + "\tnull\t" + correct_marker + "\n\n")



    f_out.close()


def format_crf_pp_file_no_pos_tags(file_name, preprocessing_obj, error_marker, correct_marker):

    f_out = open(file_name, 'wt', encoding="utf-8")
    sentences = preprocessing_obj.get_sentences()
    errors = preprocessing_obj.get_errors()
    try:

        for (sent_num, (sent, terminator)) in enumerate(sentences):
            words = preprocessing_obj.divide_sentence_into_words(sent)
            for (pos, wrd) in enumerate(words):
                if (sent_num, pos) in errors:
                    f_out.write(wrd + "\t" + error_marker + "\n")
                else:
                    f_out.write(wrd + "\t" + correct_marker + "\n")

            f_out.write(terminator + "\t" + correct_marker + "\n\n")
    except ValueError:
            print("passed")
            pass
    f_out.close()


def extract_random_sentences(preprocess_obj, percentage):

    if 1 < percentage < 0:
        print("The value of test set percentage is invalid! Exiting the system")
        sys.exit(1)

    length = len(preprocess_obj.get_sentences())
    test_size = math.ceil(length * percentage)

    random.seed(10)
    sentences = list()
    for i in range(0, test_size):
        sentences.append(preprocess_obj.pop_sentence(random.randint(0, length - 1)))
        length -= 1

    return Preprocessor(sentences)

if __name__ == "__main__":

    # initialisation
    corpus_mode = None; source = None; destination = None; pos_tagging_mode = None; crf_template_file = None
    crf_train_file = None; a = None; c = None; f = None; crf_model_file = None; crf_test_file = None;
    crf_test_with_prob = None; crf_result_file = None; corpus_training_sentences = None; corpus_test_sentences = None;
    corpus_vocab = None; corpus_training_errors = None; corpus_test_errors = None; uncertain_file = None;
    crf_uncertainty_threshold = None; error_label = None; correct_label = None; training_error_every = None;
    testing_error_every = None; percentage_of_test_set = None
    #
    # Read configuration file
    config_file = open("config.cfg", "rt", encoding="utf-8")
    for line in config_file:
        # If comment or empty line in the configuration file then skip
        if re.match('#', line) or re.match('\s*$', line):
            continue
        # Set variables
        [var, value] = line.split('=')
        if re.match('CORPUS_MODE$', var):
            corpus_mode = value.strip()
        elif re.match('SOURCE$', var):
            source = value.strip()
        elif re.match('DESTINATION$', var):
            destination = value.strip()
        elif re.match('POS_TAGGING_MODE$', var):
            pos_tagging_mode = var.strip()
        elif re.match('CRF_TEMPLATE_FILE$', var):
            crf_template_file = value.strip()
        elif re.match('CRF_TRAIN_FILE$', var):
            crf_train_file = value.strip()
        elif re.match('CRF_TRAIN_FILE_PARAM_A$', var):
            a = value.strip()
        elif re.match('CRF_TRAIN_FILE_PARAM_C$', var):
            c = value.strip()
        elif re.match('CRF_TRAIN_FILE_PARAM_F$', var):
            f = value.strip()
        elif re.match('CRF_MODEL_FILE$', var):
            crf_model_file = value.strip()
        elif re.match('CRF_TEST_FILE$', var):
            crf_test_file = value.strip()
        elif re.match('CRF_TEST_WITH_PROBABILITIES$', var):
            crf_test_with_prob = value.strip()
        elif re.match('CRF_RESULT_FILE$', var):
            crf_result_file = value.strip()
        elif re.match('CORPUS_TRAINING_SENTENCES_FILE$', var):
            corpus_training_sentences = value.strip()
        elif re.match('CORPUS_TEST_SENTENCES_FILE$', var):
            corpus_test_sentences = value.strip()
        elif re.match('CORPUS_VOCABULARY_FILE$', var):
            corpus_vocab = value.strip()
        elif re.match('CORPUS_TRAINING_ERRORS_FILE$', var):
            corpus_training_errors = value.strip()
        elif re.match('CORPUS_TEST_ERRORS_FILE$', var):
            corpus_test_errors = value.strip()
        elif re.match('CRF_UNCERTAIN_LABELS_FILE$', var):
            uncertain_file = value.strip()
        elif re.match('CRF_UNCERTAINTY_THRESHOLD$', var):
            crf_uncertainty_threshold = value.strip()
        elif re.match('SPELLING_ERROR_LABEL$', var):
            error_label = value.strip()
        elif re.match('CORRECT_SPELLING_LABEL$', var):
            correct_label = value.strip()
        elif re.match('TRAINING_ERRORS_IN_EVERY$', var):
            training_error_every = value.strip()
        elif re.match('TESTING_ERRORS_IN_EVERY$', var):
            testing_error_every = value.strip()
        elif re.match('PERCENTAGE_OF_TEST_SET$', var):
            percentage_of_test_set = value.strip()

    config_file.close()


    # delete previous files
    print("Deleting previous output files.")
    if os.path.exists(destination):
     shutil.rmtree(destination)
    os.mkdir(destination)

    # Build object
    p = Preprocessor()
    if re.match("[xX][mM][lL]$", corpus_mode):                                  # If xml mode is triggered
        p.set_xml_path_and_corpus_path(source, destination)
        p.remove_xml_tags()
    elif re.match("[pP][lL][aA][iI][nN]$", corpus_mode):                        # If plain mode is triggered
        p.set_corpus_path(source)
    else:
        print("Unknown corpus mode! Corpus mode has to be either 'XML' or 'PLAIN'. Exiting the system")
        sys.exit(1)

    p.load_corpus_sentences()                                                   # load corpus
    print ("Number of sentences in general = " + str(len(p.get_sentences())))



    # Match a percentage value
    if re.match('[0]\.[0-9][0-9]*', percentage_of_test_set):
        # Create the test set by extracting some sentences from 'p' object.
        test_obj = extract_random_sentences(p, float(percentage_of_test_set))
        test_obj.set_words_list(p.get_words_list())
    else:
        print("The percentage of the test set is unacceptable! Exiting the system")
        sys.exit(1)

    # Build the vocabulary of the training set object. This vocabulary will be used to create errors in
    # both the training and test sets.
    p.build_vocabulary()
    if re.match('[0-9]+', training_error_every) and re.match('[0-9]+', testing_error_every):
        training_error_every = int(training_error_every)
        testing_error_every = int(testing_error_every)
        p.put_errors_in_n_words_from_list(p.get_vocabulary())
        #test_obj.put_errors_in_n_words_with_distance_from_list(p.get_vocabulary())
        test_obj.put_errors_in_n_words_from_list(p.get_vocabulary())
    else:
        print("The value of the variable 'ERRORS_IN_EVERY' is unacceptable! It has to be an integer number. "
              "Exiting the system")
        sys.exit(1)

    # Write the training sentences and errors to the files if the files are given
    if corpus_training_sentences:
        p.write_sentences(corpus_training_sentences)
    if corpus_training_errors:
        p.write_errors(corpus_training_errors)

    # Write the test sentences and errors to the files if the files are given
    if corpus_test_sentences:
        test_obj.write_sentences(corpus_test_sentences)
    if corpus_test_errors:
        test_obj.write_errors(corpus_test_errors)

    crf = CRFPlusPlusInterface(error_label)                                     # Create the CRF++ object
    # Set the CRF++ parameters
    if re.match('CRF-L1$', a):
        a = 'CRF-L1'
    elif re.match('CRF-L2$', a):
        a = 'CRF-L2'
    else:
        print("Unknown CRF++ 'a' option, setting 'a' to the default value: CRF-L2")
        a = 'CRF-L2'

    # Match float or int
    if re.match('[0-9]+(\.[0-9])?[0-9]*', c):
        c = float(c)
    else:
        c = 1
        print("Unknown CRF++ 'c' option, setting it to the default value: 1.0")
    if re.match('[0-9]+', f):                                       # match integer value
        f = int(f)
    else:
        f = 1
        print("Unknown CRF++ 'f' option, setting it to the default value: 1")

    # Format the training and testing sentences to be fed to the CRF++
    #format_crf_pp_file_semantic_features(crf_train_file, p, error_label, correct_label)
    #format_crf_pp_file_semantic_features(crf_test_file, test_obj, error_label, correct_label)
    format_crf_pp_file_no_pos_tags(crf_train_file, p, error_label, correct_label)
    format_crf_pp_file_no_pos_tags(crf_test_file, test_obj, error_label, correct_label)
    # Train CRF++
    if os.path.isfile(crf_template_file) or os.path.isfile(crf_train_file) or os.path.isfile(crf_test_file):
        crf.train(crf_template_file, crf_train_file, crf_model_file, a, c, f)
    else:
        print("Training, test or template file is missing! Exiting the system")
        sys.exit(1)
    # Test CRF++
    if re.match('[tT][rR][uU][eU]$', crf_test_with_prob):
        crf.test(crf_test_file, crf_result_file, crf_model_file, True)
        if re.match('[0]\.[5-9][0-9]*', crf_uncertainty_threshold):
            uncertainty = float(crf_uncertainty_threshold)
            crf.write_lines_for_tokens_with_assignment_less_than(uncertainty, uncertain_file)
        else:
            print("The value of the 'CRF_UNCERTAINTY_THRESHOLD' parameter is unacceptable! A file with uncertainty "
                  "labeling will not be generated.")
    else:
        crf.test(crf_test_file, crf_result_file, crf_model_file)
    # Show results
    results = open("final_result.txt", 'a+', encoding="utf-8")

    results.write( "\n" + "-------------------------------------------------------" )
    results.write( "\n" + "Correct Detections: " + str(crf.get_correct_detections()))
    results.write("\n" + "Incorrect Detections: " + str(crf.get_incorrect_detections()))
    results.write("\n" + "Total number of errors in the test set: " + str(crf.get_total_errors()))
    results.write("\n" + "Undetected errors: " + str(crf.get_total_errors() - crf.get_correct_detections()))
    crf.compute_results()

