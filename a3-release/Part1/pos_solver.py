###################################
# CS B551 Fall 2023, Assignment #3
#
# Your names and user ids: Prinston Rebello (prebello), Bhanu Prakash N (bhnaraya), Dilip Nikhil Francies (dfranci)
#


from collections import defaultdict
import random
import math
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def __init__(self):
        self.all_labels = set()
        self.initial_probs = defaultdict(lambda: defaultdict(int))
        self.simple_model_prob = defaultdict(lambda: defaultdict(int))
        self.transition_prob = defaultdict(lambda: defaultdict(int))
        self.emission_prob = defaultdict(lambda: defaultdict(int))

    def simple_posterior(self, sentence, label):
        log_posterior = 0.0
        log_posterior += math.log(self.initial_probs[label[0]])
        for i in range(1, len(sentence)):
            word = sentence[i]
            current_label = label[i]
            log_posterior += math.log(self.simple_model_prob[current_label][word] + 10e-10) 

        return log_posterior
    
    def hmm_posterior(self, sentence, label):
        log_posterior = 0.0
        log_posterior += math.log(self.initial_probs[label[0]])
        for i in range(1, len(sentence)):
            word = sentence[i]
            current_label = label[i]
            prev_label = label[i-1]
            log_posterior += math.log(self.transition_prob[prev_label][current_label] + 10e-10)
            log_posterior += math.log(self.emission_prob[current_label][word] + 10e-10)

        return log_posterior

    def posterior(self, model, sentence, label):
        if model == "Simple":
            return self.simple_posterior(sentence, label)
        elif model == "HMM":
            return self.hmm_posterior(sentence, label)
        else:
            print("Unknown algo!")
 
    # Do the training!
    #
    def calculate_emission_probabilities(self, counts, vocabulary):
        for label, inner_counts in counts.items():
            total_count = sum(inner_counts.values()) + len(vocabulary)
            for word, count in inner_counts.items():
                self.emission_prob[label][word] = (count + 1) / total_count  

    def calculate_transition_probabilities(self, counts, labels):
        for prev_label, inner_counts in counts.items():
            total_count = sum(inner_counts.values()) + len(labels)
            for label, count in inner_counts.items():
                self.transition_prob[prev_label][label] = (count + 1) / total_count

    def calculate_simplified_probabilities(self, counts):
        total_word_count = sum(sum(inner_counts.values()) for inner_counts in counts.values())        
        for label, inner_counts in counts.items():
            for word, count in inner_counts.items():
                self.simple_model_prob[label][word] = count / total_word_count if total_word_count > 0 else 0.0

    def train(self, data):
        unique_words = set()
        unique_labels = set()
        simple_model_counts = defaultdict(lambda: defaultdict(int))
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))
        for sentence, labels in data:
            for i in range(len(sentence)):
                word = sentence[i]
                label = labels[i]
                simple_model_counts[label][word] += 1
                if i > 0:
                    prev_label = labels[i-1]
                    transition_counts[prev_label][label] += 1
                emission_counts[label][word] += 1
                self.all_labels.add(label)
                unique_labels.add(label)
                unique_words.add(word)
        
        start_counts = {label: 0 for label in self.all_labels}
        for sentence, labels in data:
            start_counts[labels[0]] += 1
        total_sentences = len(data)
        self.initial_probs = {label: count / total_sentences for label, count in start_counts.items()}

        self.calculate_emission_probabilities(emission_counts, unique_words)
        self.calculate_transition_probabilities(transition_counts, unique_labels)
        self.calculate_simplified_probabilities(simple_model_counts)
        

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        labels = []
        for i in range(len(sentence)):
            word = sentence[i]
            max_label = max(self.simple_model_prob.keys(), key=lambda label: self.simple_model_prob[label][word])
            labels.append(max_label)
        return labels

    # Reference: https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/
    def hmm_viterbi(self, sentence):
        dp = [{} for _ in range(len(sentence))]
        backpointer = [{} for _ in range(len(sentence))]

        for label in self.all_labels:
            dp[0][label] = math.log(self.initial_probs[label]) + math.log(self.emission_prob[label].get(sentence[0], 1e-10) + 1e-10)
            backpointer[0][label] = None

        for t in range(1, len(sentence)):
            for label in self.all_labels:
                max_prob = float('-inf')
                max_prev_label = None
                for prev_label in self.all_labels:
                    transition_prob = math.log(self.transition_prob.get(prev_label, {}).get(label, 1e-10))
                    emission_prob = math.log(self.emission_prob[label].get(sentence[t], 1e-10) + 1e-10)
                    prev_prob = dp[t - 1][prev_label]
                    current_prob = prev_prob + transition_prob + emission_prob
                    if current_prob > max_prob:
                        max_prob = current_prob
                        max_prev_label = prev_label
                dp[t][label] = max_prob
                backpointer[t][label] = max_prev_label

        best_sequence = [None] * len(sentence)
        max_final_prob = max(dp[-1].values())
        final_label = [label for label, prob in dp[-1].items() if prob == max_final_prob][0]
        best_sequence[-1] = final_label
        for t in range(len(sentence) - 2, -1, -1):
            best_sequence[t] = backpointer[t + 1][best_sequence[t + 1]]

        return best_sequence


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

