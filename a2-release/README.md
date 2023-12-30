# bhnaraya-dfranci-prebello-a2

#  Part 2 
## Truth be told:

## As a specific use case for this assignment, we’ve given you a dataset of user-generated reviews.User-generated reviews are transforming competition in the hospitality industry, because they are valuable for both the guest and the hotel owner. The task it to predict if a particular review is either "deceptive" or "truthful" using bayes network.

### Strategy and code explaination:

#### Text Pre-processing:
- **Stemming:** Stemming is a process where we extract the root form of a word. For instance, the root word of "running","eating",successful,"families" are "run","eat",success","family". We have defined a custom helper function to deal with such words. We do not want to have a large vocab of words that mean the samething which only makes the model and computataion complex.

- **Removing Punctuations, Special characters:** Any token that has special characters such as "#$&*" add no meaning to the review, atleast for computing the probability using Naive Bayes. Hence, we have removed them.

- **Stop Words:**: Any token that has a length of less than 3 letters, and are frequently used words such as " a , an, the, actually, like, b, ey, f2, have..etc is removed as they dont carry a lot of weight when calculating the probabilites.


#### Naive Bayes Model:

- A Naive bayes model consists of calculating four Probabilites:
1. P(A/B) : the Posterior probability of a class given its review.
2. P(B) : The prior probability of class
3. P(B/A) : The likelihood, which is the probability of the predictor given its class.
4. P(A) : This is the prior probability of the predictor.

Once we calculate these probabilites we use the Naive Bayes equation to find the posterior probability for each class, and the class with the highest posterior probaability is finally returned as the prediction for that review.

There are certain assumptions that are made when using Bayes network for text classification such as:

1. We assume that each feature[token] are independent of each other and all carry the same weight in predicting which class it belongs to, which is technically not eh case.

Bayes theorem is given by:

P(class/w1,w2...wn) = (P(w1,w2...wn/class) * P(class) ) / P(w1,w2,w3....wn).

Once we calculate these probabilites for the two classes that is present in the train data, we just use the odds ration and threshold to finally assign which class a particular review belongs to.

### Helper functions:

1. load_file(filename) :This is used to load the train and text file and split them into reviews, the class each review belongs to, and the number of classes in the dataset.

2. stemming(text): A custom helper function with multiple rules for based on suffixes to extract the root words.

3. remove_punctuation_numbers_specialChar(text) : removes punctuations, numbers, special characters from after tokenizing every review.

4. remove_stopwords(text, stopwords): removes stopwords for the data corups.

5. calculate_frequency(train_data) : initiates two empty dictionaries to store word and class frequency.Iterate through the training data, and check if the label is in the class frequency.  If not, initialize the class frequency for that label to 0 and also initialize an empty dictionary for word frequencies. Then increment the class frequency by 1 returning you the number of times each class appears in the entire training set.

6. classifier(train_data, test_data): In addition to the above mentioned functions, this function returns the probability of a review belonging to one of the classes present in the dataset. 

Firstly the most_prob_class and highest_score are initialized to None and negative infinityrespectively.
Then we Loop through Test Data: For each label (class) in test_data["classes"], it calculates a score for that class. This score is used to estimate the probability that the review belongs to that particular class.
The class probability is calculated based on the prior probability of the class, which was determined using the training data.
For each word in the review, it calculates the likelihood probability that the word appears in the current class.then it checks if the label exists in word_frequency and if the word exists. If yes, it gets the word count for that word in the given class, asnd sets it to zero if not found.
I have used laplace smoothing to calculate the likelihood probability.

Finally, 
If the calculated score for the current class is higher than the highest_score seen so far, the highest_score is updated, and the most_prob_class is set to the current class and the class with the highest score (most_prob_class) is assigned to the current review, and finally returns the function returns the predictionpredicted class labels for that review

