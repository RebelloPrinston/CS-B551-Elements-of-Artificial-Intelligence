#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: dfranci prebello bhanaraya [Dilip Nikhil Francies, Prinston Rebello, Bhanuprakash N]


from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import sys
import math
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

all_characters = set()
transition_prob = defaultdict(lambda: defaultdict(int))
initial_probs = defaultdict(int)
emission_prob = defaultdict(lambda: defaultdict(int))

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH) for y in range(0, CHARACTER_HEIGHT)], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    letters= { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }
    return letters

def calculate_emission_probabilities(train_letters, test_letters):
    emission_prob = {}
    for test_char_index, test_char in enumerate(test_letters):
        emission_prob[test_char_index] = {}
        for train_char_index, train_char in train_letters.items():
            # initialize the match counters
            black_match = 0
            white_match = 0
            black_mismatch = 0
            white_mismatch = 0
            # over each pixel 
            for pixel_index in range(len(test_char)):
                # matches and mismatches
                if train_char[pixel_index] == '*' and test_char[pixel_index] == train_char[pixel_index]:
                    black_match += 1
                elif train_char[pixel_index] == '*':
                    black_mismatch += 1
                elif train_char[pixel_index] == ' ' and test_char[pixel_index] == train_char[pixel_index]:
                    white_match += 1
                elif train_char[pixel_index] == ' ':
                    white_mismatch += 1
            emission_prob[test_char_index][train_char_index] = math.pow(0.9999, black_match) * math.pow(0.7, white_match) * math.pow(0.3, black_mismatch) * math.pow(0.0001, white_mismatch)

    return emission_prob

def calculate_probabilities(train_txt_fname):
    with open(train_txt_fname, 'r') as f:
        lines = f.readlines()

    simp_prob = calculate_simplified_probabilities(lines)
    transition_prob = calculate_transition_probabilities(lines)
    return simp_prob, transition_prob


def calculate_transition_probabilities(text_lines):
    for line in text_lines:
        chars = list(" ".join(line.split()))
        if chars:
            for index in range(1, len(chars)):
                transition_prob[chars[index - 1]][chars[index]] += 1

    # normalize probabilities
    for char in transition_prob:
        total = sum(transition_prob[char].values())
        for next_char in transition_prob[char]:
            transition_prob[char][next_char] /= total

    return transition_prob

def calculate_simplified_probabilities(text_lines):
    for line in text_lines:
        chars = list(" ".join(line.split()))
        if chars:
            initial_probs[chars[0]] += 1

    total = sum(initial_probs.values())
    for char in initial_probs:
        initial_probs[char] /= total

    return initial_probs


def hmm_viterbi(test_letters, train_letters, init_prob, transition_prob, emission_prob):
    dp = [{} for _ in range(len(test_letters))]
    backpointer = [{} for _ in range(len(test_letters))]

    for cur_char in train_letters:
        # dp[0][cur_char] = -math.log(emission_prob[0][cur_char]) - math.log(init_prob.get(cur_char, math.pow(10, -10))) # log 0 = negative inf so 10**-10 
        dp[0][cur_char] = math.log(emission_prob[0][cur_char]+ 10e-100) + math.log(init_prob.get(cur_char, math.pow(10, -10)))
        backpointer[0][cur_char] = [cur_char]

    for i in range(1, len(test_letters)):
        for cur_char in train_letters:
            max_prob = float('-inf')
            max_sequence = None
            for pre_char in train_letters:
                transition_prob_val = math.log(transition_prob.get(pre_char, {}).get(cur_char, math.pow(10, -10))) 
                prev_prob = dp[i - 1][pre_char]
                current_prob = prev_prob + transition_prob_val
                if current_prob > max_prob:
                    max_prob = current_prob
                    max_sequence = backpointer[i - 1][pre_char] + [cur_char]
            # dp[i][cur_char] = max_prob - math.log(emission_prob[i][cur_char])
            dp[i][cur_char] = max_prob + math.log(emission_prob[i][cur_char]+ 10e-100)
            backpointer[i][cur_char] = max_sequence

    best_sequence = [None] * len(test_letters)
    max_final_prob = max(dp[-1].values())
    final_label = [label for label, prob in dp[-1].items() if prob == max_final_prob][0]
    best_sequence = backpointer[-1][final_label]

    return ''.join(best_sequence)


def simplified(calculate_emission_probabilities_result):
    simplified_result = ""
    
    emission_prob = calculate_emission_probabilities_result
    for letter in emission_prob:
        max_emission_char = max(emission_prob[letter], key=emission_prob[letter].get)
        simplified_result += max_emission_char
    return simplified_result

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname)  = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

# train_data = load_train(train_txt_fname)
# print("\n".join([ r for r in train_letters['a'] ]))

#print("\n".join([ r for r in test_letters[0] ]))

calculate_simplified_probabilities_result, calculate_transition_probabilities_result = calculate_probabilities(train_txt_fname)
calculate_emission_probabilities_result= calculate_emission_probabilities(train_letters, test_letters)

# The final two lines of your new_string should look something like this:
print("Simple: " + simplified(calculate_emission_probabilities_result))

print("   HMM: " + ''.join(hmm_viterbi(test_letters, train_letters,calculate_simplified_probabilities_result,calculate_transition_probabilities_result, calculate_emission_probabilities_result)) )

### Reference: https://github.com/MahsaMonshizade/EL_AI/blob/main/a3/README.md