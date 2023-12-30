# Part-of-Speech Tagging

## Problem Formulation
The problem is to assign the correct part-of-speech tag to each word in a given sentence. This is a crucial task in Natural Language Processing (NLP) and forms the basis for many other tasks such as Sentiment Analysis, etc.

##  Working
The program uses two models to solve the problem: a Simplified model and a Hidden Markov Model (HMM).

### Simplified Model
In the Simplified model, each word is assigned the tag that is most likely for that word. This is done by calculating the maximum likelihood estimate of each tag for a given word.

### Hidden Markov Model
The HMM takes into account not only the probability of a word given a tag (emission probability), but also the probability of a tag given the previous tag (transition probability). The Viterbi algorithm is used to find the most likely sequence of tags for a given sequence of words.

### Probabilities

1. **Initial Probabilities**: The initial probability of a tag is calculated as the number of times the tag appears as the first tag in a sentence divided by the total number of sentences.

    $$P(tag) = \frac{\text{Number of times tag is first in a sentence}}{\text{Total number of sentences}}$$

2. **Transition Probabilities**: The transition probability from tag A to tag B is calculated as the number of times tag B follows tag A divided by the total number of times tag A appears.

    $$P(B|A) = \frac{\text{Number of times B follows A}}{\text{Total number of times A appears}}$$

3. **Emission Probabilities**: The emission probability of a word given a tag is calculated as the number of times the word is tagged with the tag divided by the total number of times the tag appears.

    $$P(word|tag) = \frac{\text{Number of times word is tagged with tag}}{\text{Total number of times tag appears}}$$

## Problems, Assumptions, Simplifications, and Design Decisions
- **Problems**: The main challenge in POS tagging is dealing with words that have not been encountered in the training data. These are known as out-of-vocabulary (OOV) words. 
- **Assumptions**: The program assumes that the sentences are independent of each other. This means that the start of a new sentence does not depend on the previous sentence.
- **Simplifications**: The program simplifies the problem by treating it as a supervised learning problem. It uses a labeled dataset to train the models.
- **Design Decisions**: The program uses log probabilities instead of raw probabilities to avoid underflow issues. It also uses smoothing to handle OOV words.

## Team Contributions

### Work Division
The team adopted a collaborative approach to the assignment. Each member was assigned a specific aspect of the project to research and develop. This allowed for a division of labor that capitalized on the strengths and interests of each team member. The team then came together to integrate their individual contributions into a cohesive whole.

### Individual Contributions
- **Bhanu Prakash N (bhnaraya)**: Bhanu was responsible for researching Hidden Markov Models (HMMs) for Part-of-Speech (POS) tagging. This involved understanding the theoretical underpinnings of HMMs and how they can be applied to the problem of POS tagging.
- **Prinston Rebello (prebello)**: Prinston focused on developing the HMM using dynamic programming. This involved translating the theoretical model researched by Bhanu into a practical algorithm that could be implemented in code.
- **Dilip Nikhil Francies (dfranci)**: Dilip was in charge of calculating the probabilities . This involved understanding and implementing the mathematical calculations necessary for the operation of our models.

After each team member completed their individual tasks, the team came together to integrate their work. we carefully review of each component, so that they worked together. We then worked together to debug any issues and optimize the code.
The result was a robust and efficient solution to the POS tagging problem.


# Optical Character Recognition

## Command- 
python3 image2text.py courier-train.png bc.train test_images/test-1-0.png
courier-train.png is the train-image-file
bc.train is the train-text
test images taken from test_image folder


## Problem Formulation
The problem is to recognize characters from an image. This is a crucial task in the field of computer vision and has applications in various domains such as reading scanned documents, license plate detection, etc.

## Working
The program uses two models to solve the problem: a Simplified model and a Hidden Markov Model (HMM).

### Simplified Model
In the Simplified model, each character is recognized independently of the others. The character that has the maximum emission probability given the observed data is chosen as the recognized character.

### Hidden Markov Model
The HMM takes into account not only the emission probabilities but also the transition probabilities between characters. The Viterbi algorithm is used to find the most likely sequence of characters given the observed data.

### Probabilities

1. **Transition Probabilities**: The transition probability from character A to character B is calculated as the number of times character B follows character A divided by the total number of times character A appears. This is done for each pair of characters in the training text.

    $$P(B|A) = \frac{\text{Number of times B follows A}}{\text{Total number of times A appears}}$$

2. **Emission Probabilities**: The emission probability of a test character given a training character is calculated using a match/mismatch model. For each pixel, if the pixel in the test character matches the corresponding pixel in the training character, the match probability is multiplied by a high probability (0.9999 for black pixels and 0.7 for white pixels). If the pixels do not match, the mismatch probability is multiplied by a low probability (0.0001 for black pixels and 0.3 for white pixels). The final emission probability is the product of all the match/mismatch probabilities for all the pixels.

    $$P(test\_char|train\_char) = \prod_{i=1}^{n} P(pixel_i|train\_char)$$

    where $P(pixel_i|train\_char)$ is either the match or mismatch probability depending on whether the pixel in the test character matches the corresponding pixel in the training character which leads to the m value mentioned in the question.

## Problems, Assumptions, Simplifications, and Design Decisions
- **Problems**: The main challenge in OCR is dealing with variations in fonts, sizes, and distortions in the characters.
- **Assumptions**: The program assumes that the characters are of a fixed size and are neatly aligned. It also assumes that the characters are black and the background is white.
- **Simplifications**: The program simplifies the problem by converting the images into binary format (black and white) and by considering only a limited set of characters.
- **Design Decisions**: The program uses log probabilities instead of raw probabilities to avoid underflow issues. It also uses a sliding window approach to extract characters from the image.

# Team Contributions

The team adopted a collaborative approach similar to the one used in the Part-of-Speech (POS) tagging problem. After laying the foundation, the team worked together to adapt the existing HMM model from the POS tagging problem to the OCR problem. This involved a significant change in how the emission probabilities were calculated, switching from a word/tag model to a match/mismatch model based on pixels. This change was a game changer and required a lot of research and adaptation by all team members. The result was a robust and efficient solution to the OCR problem.
