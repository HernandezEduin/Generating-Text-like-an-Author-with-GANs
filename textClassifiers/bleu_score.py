# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:37:41 2022

@author: Eduin Hernandez
"""
import argparse

import nltk
import random
from nltk.translate.bleu_score import SmoothingFunction

from authors import authors as authors_dict


def get_tokenlized(file):
    """tokenlize the file"""
    tokenlized = list()
    with open(file, encoding='utf-8') as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            if len(text) < 5:
                continue
            tokenlized.append(text)
    return tokenlized

class BLEU():
    def __init__(self, name=None, test_text=None, real_text=None, gram=3, portion=1):
        assert type(gram) == int or type(gram) == list, 'Gram format error!'
        self.test_text = test_text
        self.real_text = real_text
        self.gram = [gram] if type(gram) == int else gram
        self.sample_size = 200  # BLEU scores remain nearly unchanged for self.sample_size >= 200
        self.reference = None
        self.is_first = True
        self.portion = portion  # how many portions to use in the evaluation, default to use the whole test dataset

    def get_score(self, given_gram=None):
        """
        Get BLEU scores.
        :param is_fast: Fast mode
        :param given_gram: Calculate specific n-gram BLEU score
        """
        
        if self.is_first:
            self.get_reference()
            self.is_first = False
        return self.get_bleu(given_gram)

    def reset(self, test_text=None, real_text=None):
        self.test_text = test_text if test_text else self.test_text
        self.real_text = real_text if real_text else self.real_text

    def get_reference(self):
        reference = self.real_text.copy()

        # randomly choose a portion of test data
        # In-place shuffle
        random.shuffle(reference)
        len_ref = len(reference)
        reference = reference[:int(self.portion * len_ref)]
        self.reference = reference
        return reference

    def get_bleu(self, given_gram=None):
        if given_gram is not None:  # for single gram
            bleu = list()
            reference = self.get_reference()
            weight = tuple((1. / given_gram for _ in range(given_gram)))
            for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
                bleu.append(self.cal_bleu(reference, hypothesis, weight))
            return round(sum(bleu) / len(bleu), 3)
        else:  # for multiple gram
            all_bleu = []
            for ngram in self.gram:
                bleu = list()
                reference = self.get_reference()
                weight = tuple((1. / ngram for _ in range(ngram)))
                for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
                    bleu.append(self.cal_bleu(reference, hypothesis, weight))
                all_bleu.append(round(sum(bleu) / len(bleu), 3))
            return all_bleu

    @staticmethod
    def cal_bleu(reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

def str2bool(string):
    if isinstance(string, bool):
       return string
   
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Author Text Bleu Metric')

    parser.add_argument('--test-filepath', type=str, default="../textGAN/samples/dickens_test_seqgan.txt", help='Test Text to Evaluate')
    parser.add_argument('--test-author', type=str, default='dickens', help='Test Author being evaluated')
    
    parser.add_argument('--show-val-metrics', type=str2bool, default='False', help='Calculate and Show Train metrics')
    parser.add_argument('--show-test-metrics', type=str2bool, default='True', help='Calculate and Show Train metrics')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__': 
    args = parse_args()
    
    train_dataset, val_dataset, _ = authors_dict[args.test_author.lower()]
    
    train = get_tokenlized(train_dataset)

    bleu = BLEU('BLEU', gram=[2, 3, 4, 5])

    print('Author: ', args.test_author)
    if args.show_val_metrics:    
        val = get_tokenlized(val_dataset)
        bleu.reset(test_text=train, real_text=val)
        print('Val Bleu - ', bleu.get_score())
    
    if args.show_test_metrics:
        test = get_tokenlized(args.test_filepath)
        bleu.reset(test_text=train, real_text=test)
        print('Test Bleu - ', bleu.get_score())