# -*- coding: utf-8 -*-
"""
Created in 2016

@authors: Atheer Alkhalifa, Lamia Alkwai, Waleed Alsanie and Mohamed Alkanhal
         The National Center for Computation Technology & Applied Mathematics
         {aalkhalifa,lalkwai,walsanie, alkanhal} [at] kacst [dot] edu [dot] sa

This class is part of a context sensitive spell checking module. It is aimed to provide methods necessary
to preprocess the corpus and make it ready for learning.
"""


import os
import sys
import re
import random
import xml.etree.ElementTree as Et
from subprocess import *
from nltk.stem.isri import ISRIStemmer




class Preprocessor:

    def __init__(self, sentences=None):
        """
        This is a constructor of any object of this class. Object will be constructed with the list of sentences given
        in the parameter 'sentences'. This list consists of 2-tuple where each tuple represents a corpus sentence
        and its terminator as follows:
                       [(sent1, terminator1), .... ]
        If the parameter 'sentences' is not given, an object will be constructed with an empty list.
        :param sentences:
        :return:
        """

        self.__ar_sent_terminator_regex = "[.!?؟]+"
        # This is the regular expression of special words which might appear in the corpus and needed to be treated
        # as standalone words
        self.__special_words_regex = "[\\-:_~/><\"\[\]\{\}\(\)\+\*\|\'\=\&\^%\$#@`]"
        self.__regex_end_xml = "</DOC>"
        self.__xml_dir = None
        self.__corpus_dir = None
        self.__xml_file = None
        self.__corpus_file = None
        # This is a list of 2-tuple where each tuple represents a corpus sentence and its terminator
        #               [(sent1, terminator1), .... ]
        # If the parameter sentences, this attribute will be set to it. Otherwise, it will b constructed.
        if sentences:
            self.__sentences = sentences
        else:
            self.__sentences = list()

        self.__number_of_distinct_words = 0
        self.__number_of_words = 0
        # The structure of self__vocabulary is as follows:
        #           self.__vocabulary = {word1:{sentence_num:[position1,position2,.... etc]},
        #                                word2: .....
        #                               }
        # where sentences_num is the number of the sentence where the word appears, and position is the position
        # in the sentence where the word appears (a word may appear more than once in a sentence).
        self.__vocabulary = dict()
        # Errors are stored in self.__errors as follows:
        #           self._errors = {(sentence_num, position): (correct, incorrect),
        #                           .....
        #                          }
        self.__errors = dict()
        self.__words_list = list()
        # This holds the list of words that the model will learn to detect their contextual errors.
        self.__number_of_sentences = 0

    def clear_sentences(self):
        """
        This method clears all sentences. After this method is called, 'get_sentences' will return empty list
        :return: None
        """

        del self.__sentences
        self.__sentences = list()

    def clear_vocabulary(self):
        """
        This method clears the vocabulary list. After this method is called, 'get_vocabulary' will return
        empty dictionary.
        :return: None
        """

        del self.__vocabulary
        self.__vocabulary = dict()
        self.__number_of_distinct_words = 0
        self.__number_of_words = 0

    def get_sentences(self):
        """
        Returns the sentences of the corpus. The elements of this list are 2-tuple where each tuple represents a
        sentence and its terminator as follows:
                 [(sent1, terminator1), .... ]
        :return: a list of sentences
        """

        return self.__sentences

    def get_errors(self):
        """
        Returns the errors added in the corpus.
        :return: a dictionary of errors as follows:
                            {(sentence_num, position): (correct, incorrect),
                                   .....
                            }
        """

        return self.__errors

    def clear_errors(self):
        """
        This method clears all the errors that have made in the corpus.
        :return: None
        """

        del self.__errors
        self.__errors = dict()

    def get_vocabulary(self):
        """
        Returns the vocabulary built from the corpus.
        :return: The vocabulary whose structure is as follows:
        #                             = {word1:{sentence_num:[position1,position2,.... etc]},
        #                                word2: .....
        #                               }
        # where sentences_num is the number of the sentence where the word appears, and position is the position
        # in the sentence where the word appears (a word may appear more than once in a sentence).
        """

        return self.__vocabulary

    def get_number_of_words_in_vocabulary(self):
        """
        Returns the number of words in the vocabulary built from the corpus. These are the number of distinct words.
        :return: Number of words in the vocabulary
        """

        return self.__number_of_distinct_words

    def get_number_of_words(self):
        """
        Returns the number of words in the corpus. By words we mean any sequence letters.
        :return: Number of words in the corpus.
        """

        return self.__number_of_words

    def set_regex_end_xml(self, regex):
        """
        Sets the regular expression of the last tag in the xml files. Default is "</DOC>"
        :param regex: the regular expression you want to set.
        :return: None
        """

        self.__regex_end_xml = regex

    def get_regex_end_xml(self):
        """
        :return: The current regular expression of ending an xml file.
        """

        return self.__regex_end_xml

    def set_ar_sent_terminator_regex(self, regex):
        """
        Sets the regular expression of the terminators of the Arabic sentences. Default is "[\\n\\r?؟\.!]"
        :param regex: the regular expression you want to set.
        :return: None
        """

        self.__ar_sent_terminator_regex = regex

    def is_sent_terminator(self, string):
        """
        This method checks if the given 'string' is a sentence terminator. This is checked with respect to
        the regular expression set by Preprocessor.set_ar_sent_terminator_regex
        :param string: To check
        :return: True if sentence terminator, False otherwise.
        """

        return True if re.match(self.__ar_sent_terminator_regex + "$", string) else False

    def get_ar_sent_terminator_regex(self):
        """
        :return: The current regular expression of Arabic sentence terminators.
        """

        return self.__ar_sent_terminator_regex

    def set_special_words_regex(self, regex):
        """
        This method sets the regular expressions of the special words which need to be treated as standalone words.
        Default setting is "[\\-:_~/><\"\[\]\{\}\(\)\+\*\|\'\=\&\^%\$#@`]"
        :param regex: Regular expression
        :return:
        """

        self.__special_words_regex = regex

    def get_special_words_regex(self):
        """
        Returns the regular expressions of the special words.
        :return: Regular expressions of the special words.
        """

        return self.__special_words_regex

    def delete_word_from_corpus(self, word, sent_num, position):
        """
        This method deletes word 'word' in sentence number 'sent_num' at position 'position' from the corpus.
        :param word: word to be deleted.
        :param sent_num: the sentence number where the word exists.
        :param position: the position in the sentence where the word exists.
        :return: None.
        """

        list_of_sentences = self.get_sentences()
        if sent_num < 0:
            print("Cannot delete word! Sentence index < 0: unacceptable sentence index!")
            return
        if len(list_of_sentences) - 1 < sent_num:
            print("Cannot delete word! Sentence index out of range: unacceptable sentence index!")
            return
        if position < 0:
            print("Cannot delete word! word position < 0: unacceptable position!")
            return

        sent = list_of_sentences.pop(sent_num)
        sent_string = sent[0]
        sent_terminator = sent[1]
        split_into_words = self.divide_sentence_into_words(sent_string)
        if len(split_into_words) - 1 < position:
            print("Cannot delete word! word position out of range: unacceptable position!")
            return

        joint = ' '.join(split_into_words)
        if split_into_words[position] != word:
            print("Cannot delete word '" + word + "'! It does not exist in sentence number " + str(sent_num) +
                  ". This sentence has been found instead:\n" + joint)
            list_of_sentences.insert(sent_num, sent)
            return

        del split_into_words[position]
        joint = ' '.join(split_into_words)
        list_of_sentences.insert(sent_num, (joint, sent_terminator))

        # If a vocabulary list for this object has not been built, then do not update the vocabulary list
        if not self.__vocabulary:
            return

        # Otherwise, if a vocabulary list for this object has been built, then update the vocabulary list by deleting
        # this word from it
        self.__vocabulary[word][sent_num].remove(position)  # Remove the position of the word from the sentence
        if not self.__vocabulary[word][sent_num]:           # If no appearance of this word in the sentence
            del self.__vocabulary[word][sent_num]           # delete the sentence from the vocabulary entry of the word
            if not self.__vocabulary[word]:                 # If the word does not appear any further in the corpus
                del self.__vocabulary[word]
                self.__number_of_distinct_words -= 1

        if position < len(split_into_words):
            # If the word is not the last one in the sentence
            # then for every other word whose position is greater then 'position', reduce its position one place.
            for i in range(position, len(split_into_words)):
                word_after_position = i + 1
                word_after_index = self.__vocabulary[split_into_words[i]][sent_num].index(word_after_position)
                self.__vocabulary[split_into_words[i]][sent_num].remove(word_after_position)
                self.__vocabulary[split_into_words[i]][sent_num].insert(word_after_index, i)

        self.__number_of_words -= 1

    def add_word_to_corpus(self, word, sent_num, position):
        """
        This method adds word 'word' to the corpus in the sentence number 'sent_num' at position 'position'.
        :param word: word to be added.
        :param sent_num: the sentence number where the word will be added.
        :param position: the position in the sentence where the word will be added.
        :return: None.
        """

        list_of_sentences = self.get_sentences()
        if sent_num < 0:
            print("Cannot add word! Sentence index < 0: unacceptable sentence index!")
            return
        if len(list_of_sentences) - 1 < sent_num:
            print("Cannot add word! Sentence index out of range: unacceptable sentence index!")
            return
        if position < 0:
            print("Cannot add word! word position < 0: unacceptable position!")
            return

        sent = list_of_sentences.pop(sent_num)
        sent_string = sent[0]
        sent_terminator = sent[1]
        split_into_words = self.divide_sentence_into_words(sent_string)
        if len(split_into_words) < position:
            print("Cannot add word! word position out of range: unacceptable position!")
            list_of_sentences.insert(sent_num, sent)
            return

        split_into_words.insert(position, word)
        list_of_sentences.insert(sent_num, (' '.join(split_into_words), sent_terminator))

        # If a vocabulary list for this object has not been built, then do not update the vocabulary list
        if not self.__vocabulary:
            return

        # Otherwise, if a vocabulary list for this object has been built, then update the vocabulary list by adding
        # this word to it
        if word in self.__vocabulary:                                   # If the word is already in the vocabulary
            if sent_num in self.__vocabulary[word]:                     # if the sentence already exists
                self.__vocabulary[word][sent_num].append(position)
                self.__vocabulary[word][sent_num].sort()
            else:
                self.__vocabulary[word][sent_num] = [position]
        else:                                                           # If the word does not exist in vocabulary
            self.__vocabulary[word] = {sent_num: [position]}            # add it
            self.__number_of_distinct_words += 1                        # increase the number of distinct words

        if position < len(split_into_words) - 1:
            # If the word is not the last one in the sentence
            # then for every other word whose position is greater then 'position', increment its position one place.
            for i in range(position + 1, len(split_into_words)):
                word_after_position = i - 1
                word_after_index = self.__vocabulary[split_into_words[i]][sent_num].index(word_after_position)
                self.__vocabulary[split_into_words[i]][sent_num].remove(word_after_position)
                self.__vocabulary[split_into_words[i]][sent_num].insert(word_after_index, i)

        self.__number_of_words += 1

    def set_xml_path_and_corpus_path(self, xml_source, corpus=None):
        """
        This method sets the path of the xml file(s) which will processed to remove the tags and generate the corpus
        file(s). the 'xml_source' is an xml file, 'corpus' must be a file name of the corpus file to be generated,
        otherwise, when 'xml_source' is a directory containing xml files, 'corpus' is must be a directory to contain
        the corpus files generated from the xml files in the source directory:
        :param xml_source: This is the source file/directory of the xml file(s)
        :param corpus: This is the destination file/directory where the corpus file(s) will be written
                         after the xml tags are removed. If not passed, the xml_source must be a file and the
                         generated corpus file must have the same name as the source with extension "txt"
        :return: None
        """

        try:
            if not os.path.exists(xml_source):
                raise IOError(xml_source + " does not exist!")

            if os.path.isfile(xml_source):
                self.__xml_file = xml_source
                if corpus is None:  # if corpus has not been set, give it the same name as xml but "txt" instead
                    self.__corpus_file = os.path.splitext(self.__xml_file)[0] + ".txt"
                elif os.path.isfile(corpus):         # if corpus is a name of a file, set it
                    self.__corpus_file = corpus
                elif os.path.exists(corpus):         # if corpus is an existing directory
                    self.__corpus_file = corpus + os.path.basename(os.path.splitext(self.__xml_file)[0]) + ".txt"
                else:                                   # if corpus is a non-existing file, then create it
                    self.__corpus_file = corpus
            # if xml_source is a directory and corpus is not
            elif corpus is None:
                raise IOError(" You need to choose a directory in which the output files will be placed")
            elif not os.path.isdir(corpus):
                raise IOError(str(corpus) + " has to be a directory")
            else:
                self.__xml_dir = xml_source
                self.__corpus_dir = corpus
        except IOError as err:
            sys.stderr.write("ERROR: in setting the 'xml' source and 'txt' destination ")
            print(str(err), file=sys.stderr)
            sys.exit(1)

    def xml_path_is_set(self):
        """
        This method checks if xml path is set (not None).
        :return: True: if xml path is set, False otherwise. That I'm going to the this country
        """

        return (self.__xml_dir is not None) or (self.__xml_file is not None)

    def corpus_path_is_set(self):
        """
        This method checks if corpus path is set (not None).
        :return: True: if corpus path is set, False otherwise.
        """
        return (self.__corpus_dir is not None) or (self.__corpus_file is not None)

    def get_current_set_xml_path(self):
        """
        This returns the currently set path for the xml file(s).
        :return: The setting of the xml path
        """

        return self.__xml_dir if self.__xml_dir is not None else self.__xml_file

    def get_current_set_corpus_path(self):
        """
        This returns the currently set path for corpus file(s).
        :return: The setting of the corpus path
        """

        return self.__corpus_dir if self.__corpus_dir is not None else self.__corpus_file

    def set_corpus_path(self, corpus):
        """
        This method sets the path to your corpus file(s). The path can either be a directory containing file from
        which your corpus will be loaded, or a file contaning your corpus.
        :param corpus: The path to the corpus
        :return: None.
        """

        if os.path.isdir(corpus):
            self.__corpus_dir = corpus
            self.__corpus_file = None
        elif os.path.isfile(corpus):
            self.__corpus_file = corpus
            self.__corpus_dir = None
        else:
            print("ERROR: Cannot set corpus path! Directory or file does not exist!")
            sys.exit(1)

    def remove_xml_tags(self):
        """
        This method is passed either a file or a directory. If it is passed a directory, it removes the xml tags
        in the files in that directory, otherwise, it removes the xml tags in the file. If the destination is
        given the resulting file(s) are written in the destination.
        :return: None
        """

        if not self.xml_path_is_set():
            print("You need to set the path of the directory where your xml files are placed or the " +
                  "path of your xml file.")
            sys.exit(1)

        if self.__xml_dir is None:
            self.__remove_xml_tags_file(self.__xml_file, self.__corpus_file)
        else:
            corpus_dir = self.__corpus_dir
            xml_dir = self.__xml_dir
            if not self.__corpus_dir.endswith(os.sep):        # if the directory path does not end with slash
                corpus_dir += os.sep
            if not self.__xml_dir.endswith(os.sep):         # if the directory path does not end with slash
                xml_dir += os.sep
            for file in os.listdir(xml_dir):         # remove xml tags in all the files in the directory
                xml_file_name = xml_dir + file
                corpus_file_name = corpus_dir + os.path.splitext(os.path.basename(xml_file_name))[0] + ".txt"
                self.__remove_xml_tags_file(xml_file_name, corpus_file_name)

        print("Finished removing XML tags successfully.")

    def __remove_xml_tags_file(self, xml_file, corpus_file):
        """
        This is a private method and should not be called from outside the class. If you want to remove
        the xml tags, please use the public method 'remove_xml_tags'
        :param xml_file: a file containing xml documents
        :param corpus_file: a file to which the corpus will be written after the xml tags have been removed
        :return: None
        """

        print("Processing file " + xml_file + " to remove XML tags ....")
        input_f = open(xml_file, 'rt', encoding="utf-8")
        txt = ""
        final_text = ""
        end_xml = re.compile(self.__regex_end_xml)
        try:                                        # Catch errors resulting from reading non-text (xml) files
            for line in input_f:
                if end_xml.search(line) is None:
                    txt += line
                else:
                    try:
                        txt += line
                        tree = Et.fromstring(txt)
                        final_text += Et.tostring(tree, encoding='unicode', method='text')
                    except Et.ParseError as p_err:
                        print(format(p_err), file=sys.stderr)
                        print("ERROR: In parsing XML file " + xml_file, file=sys.stderr)
                        input_f.close()
                        sys.exit(1)
                    else:
                        txt = ""
        except UnicodeDecodeError:
            print("ERROR: Cannot read file: " + xml_file + ". It is not a readable text file!")
            input_f.close()
            return

        input_f.close()
        out_f = open(corpus_file, 'wt', encoding="utf-8")
        out_f.write(final_text)
        out_f.close()

    def load_corpus_sentences(self):
        """
        This method loads the sentences of the corpus from the source. If the source is a directory, it processes all
        readable files in this directory. If the source is a file, it process the corpus in this file.
        Any previous loaded sentences will be cleared, and the sentences of the corpus will be the loaded ones.
        :return: None
        """

        self.clear_sentences()

        if not self.corpus_path_is_set():
            print("You need to set the path of the directory where your corpus files are placed " +
                  "or the path of your corpus file.")
            sys.exit(1)

        if self.__corpus_dir is None:
            self.__load_file(self.__corpus_file)
        else:
            if not self.__corpus_dir.endswith(os.sep):
                self.__corpus_dir += os.sep
            for file in os.listdir(self.__corpus_dir):
                self.__load_file(self.__corpus_dir + file)

        print("Finished loading the corpus successfully.")

    def __load_file(self, file):
        """
        This is a private method and should not be called from outside the Preprocessor class. If you want to load
        your corpus, use 'load_corpus(source)'.
        :param file: The file to load text from
        :return: None
        """

        print("Loading file " + file + " ....")
        input_f = open(file, 'rt', encoding="utf-8")
        final_text = ""
        arabicL = re.compile('[\u0627-\u064a]')
        numbers = re.compile('\d')
        try:                                        # Catch errors resulting from reading non-text files
            for line in input_f:
                # remove any alphanumeric characters and single letters
                line = re.sub(r'[a-zA-Z\d\:\(\)\/\"]', ' ', line)
                line = ' '.join([w for w in line.split() if len(w) > 1])
                line = ' ' + line
                final_text += line
        except UnicodeDecodeError:
            print("ERROR: Cannot read file: " + file + ". It is not a readable text file!")
            input_f.close()
            return

        input_f.close()
        sentences = self.chop_off_text_into_sentences(final_text)
        # load sentences that have only the words in the words list
        if not self.__words_list:

            file = open("temp.txt", 'w', encoding="utf-8")
            for s in sentences[:1000]:
                print(s[0] + '/n', file=file)
            
            file.close()
            temp = check_output(
            'java    -mx1g   -cp   stanford-postagger.jar:context_sensitive_spell_chk/lib/* edu.stanford.nlp.tagger.maxent.MaxentTagger    -model    context_sensitive_spell_chk/lib/arabic.tagger    -textFile temp.txt',
            shell=True, stderr=PIPE)
            tagged_subset = temp.decode("utf-8")
            all_verbs_list = list()
            for s in tagged_subset.split(" "):
                if ("VB" in s.split("/")[1]) and len(s.split("/")[0]) > 4 and s.split("/")[0].isalpha() and not any(s.split("/")[0] in w for w in all_verbs_list):
                    if  s.split("/")[0] in [all_verbs_list]:
                        print("Duplicate $$$$$$$$$$$$$")
                    all_verbs_list.append(s.split("/")[0])

            self.__words_list = [all_verbs_list[random.randrange(len(all_verbs_list))] for item in range(100)]
        new_sentences = list()
        for (s, terminator) in sentences:
            for w in s.split(" "):
                if w in self.__words_list:
                    new_sentences.append((s, terminator))
                    break
        self.__number_of_sentences += len(new_sentences)
        print("Number of words: " + str(len(self.__words_list)))
        print ("Number of sentences " + str(len(sentences)))
        print ("Number of new sentences " + str(self.__number_of_sentences))


        #write the words in a text file
        f_out = open("words_list.txt", 'wt', encoding="utf-8")
        for w in self.__words_list:
            f_out.write(w + "\n")
        f_out.close()

        if new_sentences:
            # some list has been returned
            #self.clear_sentences()
            self.get_sentences().extend(new_sentences)



    def set_words_list(self, words_list):
        """
                This method sets the words list to from value received.
                :param words_list: A list of words to build the model upon.
                :return:
                """
        self.__words_list = words_list


    def get_words_list(self):
        """
                This method returns the words list.
                :param:
                :return: words_list: A list of words to build the model upon.
                """
        return self.__words_list


    def pop_sentence(self, sent_num):
        """
        This method pops and returns sentence number 'sent_num' from the sentences in this object.

        NOTE: It is always more efficient to use this method before you build the vocabulary. As popping a sentence,
        after building a vocabulary causes deleting all the words constituting it from the vocabulary.
        :param sent_num: The number of sentence to be popped.
        :return:
        """

        if not self.__sentences:
            return

        sentence_tuple = self.__sentences.pop(sent_num)                               # pop the sentence
        sentence = sentence_tuple[0]

        # If the vocabulary has been built, delete the words of the sentence from the vocabulary.
        if self.__vocabulary:
            words = self.divide_sentence_into_words(sentence)                       # get a list of its words
            # Put them in a set to delete them from the vocabulary. Putting them in a set,
            # is aimed to delete word which exists twice in the sentence only once
            set_words = set(words)
            # deletes its words from vocabulary
            for wrd in set_words:
                # delete the sentence entry of the word in the vocabulary
                del self.__vocabulary[wrd][sent_num]
                # if the word does not appear in any other sentence according to the vocabulary, then delete it
                if not self.__vocabulary[wrd]:
                    del self.__vocabulary[wrd]

            # update the vocabulary so that any sentence number of words appearing in a sentence whose index is
            # greater than the index of te popped sentence is reduced by one
            self.__vocabulary = {
                                    wd:     {
                                               (lambda sent: sent - 1 if sent > sent_num else sent)(sent_n): pos
                                               for sent_n, pos in self.__vocabulary[wd].items()
                                            }
                                    for wd in self.__vocabulary
                                }

        if self.__errors:
            # delete the positions of the words of the sentence from the list of errors. If an error has been made
            # in the position of any of the sentence words, these errors will be deleted and made no longer available.
            sentence_length = len(sentence)
            for pos in range(0, sentence_length):
                if (sent_num, pos) in self.__errors:
                    del self.__errors[(sent_num, pos)]

            # update the errors so that any sentence number where an error exists is
            # greater than the index of te popped sentence is reduced by one
            self.__errors = {
                ((lambda sent: sent - 1 if sent > sent_num else sent)(sent_n), pos): (correct, incorrect)
                for (sent_n, pos), (correct, incorrect) in self.__errors.items()
                            }

        return sentence_tuple

    def chop_off_text_into_sentences(self, text):
        """
        This method takes a text and divides it into sentences with respect to the separators set in
        'Preprocessor.set_ar_sent_terminator_regex(self, regex)'. The sentences will be returned as a list
        of 2-tuple elements where each tuple represents a sentence and its terminator as follows:
                                [(sent1, terminator1), ....]
        :param text: A text to split.
        :return: a list of tuples of sentences and their respected terminators.
        """

        if text.isspace() or not text:                                          # if the text is just spaces or empty
            return

        # Split sentence with respect to the terminators and keep the terminators as elements of the list
        s = re.split("(" + self.get_ar_sent_terminator_regex() + ")", text)
        # Remove empty string and spaces resulting from the splitting
        s = [word.strip() for word in s if word and not word.isspace()]

        sentence = ""
        final_list = list()
        for sent in s:
            if self.is_sent_terminator(sent):
                final_list.append((sentence, sent))
                sentence = ""
            else:
                sentence += sent

        if sentence:                                                        # If the last sentence is not terminated
            final_list.append((sentence, ""))                               # by a terminator

        return final_list

    @staticmethod
    def levenshtein(s, t):
        """
        This method computes the Levenshtein distance between two strings 's' and 't'. We have taken it as it is
        from a wikipedia article indicating courtesy of:
                        - Christopher P. Matthews,
                        - christophermatthews1985@gmail.com,
                        - Sacramento, CA, USA
        :param s: First string
        :param t: Second string
        :return: Levenshtein distance between 's' and 't'.
        """

        if s == t:
            return 0
        elif len(s) == 0:
            return len(t)
        elif len(t) == 0:
            return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]

        return v1[len(t)]

    def divide_sentence_into_words(self, text):
        """
        This method divides a sentences into a list of its constituting words. Special words set by
        'Preprocessor.set_special_words_regex(regex)' will be considered words by themselves.
        :param text: The sentence to be divided
        :return: A list of words constituting the sentence.
        """

        # Split each sentence into its constituting words. Keep the special characters (e.g. '{', '[', '-', ... etc)
        # As they will also be considered separate words in the vocabulary list.
        words_of_sent = re.split("\s|(" + self.__special_words_regex + ")", text)
        # do not add sentence terminators in the vocabulary
        return [word for word in words_of_sent if word and not word.isspace()]

    def build_vocabulary(self):
        """
        This method builds a python dictionary of vocabulary list, keeping track of each word regarding the sentences at
        which it appears and its positions each sentence.The vocabulary will include special words (characters)
        as well (e.g. '{', '[', '[', ... etc), but not the sentence terminators. Calling this method will wipe off
        any previous vocabulary that has been built.
        :return: None
        """

        self.clear_vocabulary()
        print("Building vocabulary list from the corpus ....")

        list_sentences = self.get_sentences()
        if len(list_sentences) == 0:
            print("There is an empty list of sentences! No vocabulary list has been built.")
            return

        for (sent_num, sent) in enumerate(list_sentences):
            # Split each sentence into its constituting words. Keep the special characters (e.g. '{', '[', '-', ... etc)
            # do not add sentence terminators in the vocabulary
            words_of_sent = self.divide_sentence_into_words(sent[0])
            for (position, word) in enumerate(words_of_sent):
                if word not in self.__vocabulary:
                    self.__vocabulary[word] = {sent_num: [position]}
                elif sent_num in self.__vocabulary[word]:   # If word is in vocabulary and appeared before in the same
                        self.__vocabulary[word][sent_num].append(position)      # sentence
                else:                                       # If word is in vocabulary but never appeared in this
                        self.__vocabulary[word][sent_num] = [position]          # sentence

                self.__number_of_words += 1                 # Increase the number of words

        self.__number_of_distinct_words = len(self.__vocabulary)        # Set the number of distinct words

        print("Finished building vocabulary list.")

    @staticmethod
    def build_alphabet(vocabulary):
        """
        This method builds the alphabet that is used to form the words of the given vocabulary.
        :param vocabulary: A python dictionary representing the vocabulary from which an alphabet will be built.
        :return:
        """

        alphabet = set()
        for word in vocabulary:
            alphabet = alphabet.union(word)

        return alphabet

    @staticmethod
    def edits1_in_vocabulary(word, vocabulary, alphabet):
        """
        This method returns all words that are one editing distance from the given word 'word'.
        It has been taken from Peter Novring's spell corrector system published in:
        http://norvig.com/spell-correct.html

        with a little change so that only words that exist in the given vocabulary are returned. And the possible
        words which will be returned will be built from the given alphabet.
        :param word: A word to find all words in the vocabulary which are 1 editing distance from it.
        :param vocabulary: A vocabulary from which words with 1 edit distance from 'word' will be looked up.
        :param alphabet: An alphabet which will be used to generate potential words.
        :return: A set of words in the vocabulary which are 1 editing distance from the given word.
        """

        if not alphabet:
            print("ERROR: Cannot return words with edit distance! The given alphabet is empty!")
            return
        if not vocabulary:
            print("ERROR: Cannot return words with edit distance! The given vocabulary is empty!")
            return

        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        # By us to exclude words not in vocabulary
        deletes = [elem for elem in deletes if elem in vocabulary]
        #
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        # By us to exclude words not in vocabulary and to exclude the same word resulting from
        # the transposition
        transposes = [elem for elem in transposes if elem != word and elem in vocabulary]
        #
        replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
        # By us to exclude words not in vocabulary and to exclude the same word resulting from
        # the replacement
        replaces = [elem for elem in replaces if elem != word and elem in vocabulary]
        #
        inserts = [a + c + b for a, b in splits for c in alphabet]
        inserts = [elem for elem in inserts if elem in vocabulary]
        return set(deletes + transposes + replaces + inserts)

    @staticmethod
    def edits2_in_vocabulary(word, vocabulary, alphabet):
        """
        This method returns all words that are two editing distance from the given word 'word'.
        It has been taken from Peter Novring's spell corrector system published in:
        http://norvig.com/spell-correct.html

        with a little change so that only words that exist in the given vocabulary are returned. And the possible
        words which will be returned will be built from the given alphabet.
        :param word: A word to find all words in which are 2 editing distance from it.
        :param vocabulary: A vocabulary from which words with 2 edit distance from 'word' will be looked up.
        :param alphabet: An alphabet which will be used to generate potential words.
        :return: A set of words in the vocabulary which are 2 editing distance from the given word.
        """

        return set(
                e2
                for e1 in Preprocessor.edits1_in_vocabulary(word, vocabulary, alphabet)
                for e2 in Preprocessor.edits1_in_vocabulary(e1, vocabulary, alphabet)
                if e2 in vocabulary
                  )





    def put_errors_in_n_words_from_list(self, vocabulary, d=1):
        """
        This method inserts an error in every sentence in the word from the main list of words. It provides a version
        of a correct and incorrect examples of the same word
        :param vocabulary: The vocabulary from which words will be looked up to replace other words.
        :param n: Number of words amongst which an error will be placed. Default is 500.
        :param d: The string distance between the correct word and the erroneous word. Default is 1.
        :return: None
        """

        if len(vocabulary) == 0:
            print("There are no words in the given vocabulary list! Errors cannot be put.")
            return

        if d == 1:                                  # If distance is set to 1, set called method to edit1
            distance = self.edits1_in_vocabulary
        elif d == 2:                                # If distance is set to 2, set called method to edit 2
            distance = self.edits2_in_vocabulary
        else:                                       # Otherwise, unacceptable and return
            print("The value of the distance is not acceptable! No error will be put")
            return

        # print("Introducing errors to the corpus in every " + str(n) + " words by replacing a randomly selected " +
        #      "word with another from the given vocabulary ....")

        self.clear_errors()                                             # clear any previous list of errors
        sentences = self.get_sentences()                                # get sentences

        alphabet = Preprocessor.build_alphabet(vocabulary)

        sentences_processed = 0                                         # start from the first sentence
        # initialise a list of words. This list will contain words, the sentences they are in and the position in the
        # sentences in a tuple as follows: [(word1,sent,position), ....]
        words = list()
        # This variable is the starting number of the sentences which the loop will look for a word to replace.
        sentence_offset = 0
        while sentences_processed < len(sentences):
            # get the list of words in sentence number sentences_processed
            list_of_words = self.divide_sentence_into_words(sentences[sentences_processed][0])
            # add them to the list of words in the structure described above
            words.extend([(wrd, sentences_processed, position) for (position, wrd) in enumerate(list_of_words)])
            number_of_words = len(words)
            # if the number of words reached the given parameter 'n', or no more sentences to processes,
            # Then start looking for a ward to replace.
            # get the word listed in the training words list
            try:
                # Select the first word found in the list
                # to_delete_index = next(wrd[2] for wrd in words if wrd[0] in words_list)
                # Select a word found in the list randomly
                #to_delete_index = random.choice(list(wrd[2] for wrd in words if wrd[0] in self.__words_list))
                to_delete_index = next(wrd[2] for wrd in words if wrd[0] in self.__words_list)
            except (StopIteration, IndexError) as error:
                sentences_processed += 1
                continue
            to_replace = words[to_delete_index]
            # find a list of possible replacements with distance 'd'
            possible_replacements = list(distance(to_replace[0], vocabulary, alphabet))
            # If there are replacement to this word in the vocabulary distance
            if possible_replacements:
                replacement = possible_replacements[random.randint(0, len(possible_replacements) - 1)]
                self.delete_word_from_corpus(to_replace[0], to_replace[1], to_replace[2])
                self.add_word_to_corpus(replacement, to_replace[1], to_replace[2])
                self.__errors[(to_replace[1], to_replace[2])] = (to_replace[0], replacement)
                # Refresh the list of words for processing another group
            del words
            words = list()
            # print("Finished inserting an error in sentence number : " + str(sentences_processed))
            sentences_processed += 1

        print("Finished putting errors in the corpus.")


    def put_errors_in_n_words_with_distance_from_list(self, vocabulary, n=300, d=1):
        """
        This method puts spelling errors in every 'n' words of the text. The error is made by choosing
        the least number of sentences constituting words greater than or equal to 'n', and then picking a word
        from the words' list found in these sentences and substituting with another word from the provided vocabulary
        whose distance from correct word is 'd'.
        :param vocabulary: The vocabulary from which words will be looked up to replace other words.
        :param n: Number of words amongst which an error will be placed. Default is 500.
        :param d: The string distance between the correct word and the erroneous word. Default is 1.
        :return: None
        """

        if len(vocabulary) == 0:
            print("There are no words in the given vocabulary list! Errors cannot be put.")
            return

        if d == 1:  # If distance is set to 1, set called method to edit1
            distance = self.edits1_in_vocabulary
        elif d == 2:  # If distance is set to 2, set called method to edit 2
            distance = self.edits2_in_vocabulary
        else:  # Otherwise, unacceptable and return
            print("The value of the distance is not acceptable! No error will be put")
            return

        # print("Introducing errors to the corpus in every " + str(n) + " words by replacing a randomly selected " +
        #      "word with another from the given vocabulary ....")

        self.clear_errors()  # clear any previous list of errors
        sentences = self.get_sentences()  # get sentences

        alphabet = Preprocessor.build_alphabet(vocabulary)

        sentences_processed = 0  # start from the first sentence
        # initialise a list of words. This list will contain words, the sentences they are in and the position in the
        # sentences in a tuple as follows: [(word1,sent,position), ....]
        words = list()
        # This variable is the starting number of the sentences which the loop will look for a word to replace.
        sentence_offset = 0
        while sentences_processed < len(sentences):
            # get the list of words in sentence number sentences_processed
            list_of_words = self.divide_sentence_into_words(sentences[sentences_processed][0])
            # add them to the list of words in the structure described above
            words.extend([(wrd, sentences_processed, position) for (position, wrd) in enumerate(list_of_words)])
            number_of_words = len(words)
            # if the number of words reached the given parameter 'n', or no more sentences to processes,
            # Then start looking for a ward to replace.
            if number_of_words >= n or sentences_processed == (len(sentences) - 1):
                found = False
                # while a word with a vocabulary entry with the specified distance is found, and there is still
                # words to look for then loop.
                while words and not found:
                    # look randomly for a word to replace in the list of words.
                    to_delete_index = next(wrd[2] for wrd in words if wrd[0] in self.__words_list)
                    to_replace = words[to_delete_index]
                    # find a list of possible replacements with distance 'd'
                    possible_replacements = list(distance(to_replace[0], vocabulary, alphabet))
                    # If there are replacement to this word in the vocabulary distance
                    if possible_replacements:
                        replacement = possible_replacements[random.randint(0, len(possible_replacements) - 1)]
                        # print("Replacing word '" + to_replace[0] + "' in sentence number " +
                        #      str(to_replace[1]) + " at position " + str(to_replace[2]) +
                        #      " with word '" + replacement + "'")
                        self.delete_word_from_corpus(to_replace[0], to_replace[1], to_replace[2])
                        self.add_word_to_corpus(replacement, to_replace[1], to_replace[2])
                        self.__errors[(to_replace[1], to_replace[2])] = (to_replace[0], replacement)
                        found = True
                    else:
                        # No vocabulary word exist which is with distance from this word
                        # so delete this one from a possible to_replace list and check another word.
                        del words[to_delete_index]
                # If a word cannot be found with vocabulary list entry with the specified distance in the list of
                # words built from the processed sentences.
                if not found:
                    print("Could not find a word to replace in the sentences residing between: " +
                          str(sentence_offset) + " and " + str(sentences_processed))
                # Refresh the list of words for processing another group
                del words
                words = list()
                # Move to the next block to look for another word
                sentence_offset = sentences_processed + 1

            sentences_processed += 1

        print("Finished putting errors in the corpus.")

    def write_sentences(self, file_name=sys.stdout):
        """
        This method writes the the current sentences to the given file. Each sentence is written in a separate line.
        :param file_name:The name of the file to which the sentences will be written
        :return: None
        """

        if file_name is sys.stdout:
            f_out = file_name
        else:
            f_out = open(file_name, 'wt', encoding="utf-8")

        list_sentences = self.get_sentences()
        # clean the corpus from empty sentences
        for (sent, terminator) in list_sentences:
            f_out.write(sent + terminator + '\n')
            f_out.write("###########################\n")


        if f_out is not sys.stdout:
            f_out.close()
            print("Corpus sentences have been written to " + file_name)

    def write_errors(self, file_name=sys.stdout):
        """
        this method writes the errors made by the preprocessor on the corpus to the file specified by file_name.
        :param file_name: The file to which this method writes.
        :return: None.
        """

        if file_name is sys.stdout:
            out_f = sys.stdout
        else:
            out_f = open(file_name, 'wt', encoding="utf-8")

        # Label columns
        out_f.write('{0:20}{1:20}{2:<20}{3:20}'.format('CORRECT', 'INCORRECT', 'SENTENCE', 'POSITION') + '\n')
        out_f.write('{:_<80}'.format('') + '\n')                        # Write '_' under the labels

        # Created a sorted list of (sent_num, position)
        # to print the errors sorted based on the sent_num
        sorted_list = sorted(self.__errors, key=lambda y: y[0])

        for (sent_num, position) in sorted_list:
            (correct, incorrect) = self.__errors[(sent_num, position)]
            out_f.write('{0:20}{1:20}{2:<20}{3}'.format(correct, incorrect, sent_num, position) + '\n')

        if out_f is not sys.stdout:
            out_f.close()
            print("List of errors that have been inserted to the corpus have been written to " + file_name)

    def write_corpus_vocabulary(self, file_name=sys.stdout):
        """
        this method writes the corpus vocabulary to the file specified by file_name.
        :param file_name: The file to which this method writes.
        :return: None.
        """

        if file_name is sys.stdout:
            out_f = sys.stdout
        else:
            out_f = open(file_name, 'wt', encoding="utf-8")

        # Label columns
        out_f.write('{0:20}{1:<20}{2}'.format('WORD', 'SENTENCE NO.', 'POSITIONS') + '\n')
        out_f.write('{:_<80}'.format('') + '\n')                          # Write '_' under the labels
        for word in self.__vocabulary:                                    # for each word in vocabulary
            out_f.write(word + ":")                                       # write the word
            # get the sentences on which this word appears
            # sort them so for convenient printing
            sent_num = sorted(self.__vocabulary[word].keys())
            for sent in sent_num:                                         # write the sentence numbers and the positions
                out_f.write('{0:20}{1:<20}{2}'.format('', sent, str(self.__vocabulary[word][sent])) + '\n')

            out_f.write("########################\n")                     # write this at the end of each word entry

        if out_f is not sys.stdout:
            out_f.close()
            print("Corpus vocabulary list has been written to " + file_name)

















