# SeekTruth.py : Classify text objects into two categories
#
# dfranci prebello bhanaraya [Dilip Nikhil Francies, Prinston Rebello, Bhanuprakash N]
#
import sys
import re
import math


def load_file(filename):
    objects,labels = [],[]  
    with open(filename, "r") as file:
        for doc in file:
            split_docs = doc.strip().split(' ', 1) # split as label : text 
            labels.append(split_docs[0]) 
            if len(split_docs)>1:
                objects.append(split_docs[1])
            else:
                raise Exception("No training or test data found, please add data to the labels")
            
            
    
    return {"objects": objects, "labels": labels, "classes": list(set(labels))} 


def stemming(text): # helper function to stem the words with customized rules for different kinds of words

    if text.endswith('ies'): #returns family for families
        return text[:-3] + 'y'
    
    if text.endswith('eed'):
        if text.count('e') > 1:
            return text[:-3] + 'ee'
        else:
            return text[:-3]
    
    if text.endswith('al') or text.endswith('er') :
            return text[:-2] 
        
    if text.endswith('ly'):
        if 'ly' in text:
            return text[:-2] 
            
    if text.endswith('ing') or text.endswith('ed') : #returns 
        if 'ing' in text or 'ed' in text:
            if text[-3] in "aeiou":
                return text[:-3] 
            
    if text.endswith('ize'): #returns rational for rationalize
        if 'ize' in text:
            return text[:-3]
    return text


def stemmed_text(text):
    words = text.split()
    stem_words = [stemming(word) for word in words] # call the helper function to stem all the tokens
    return ' '.join(stem_words)

def classifier(train_data, test_data):  #returns the predictions of each review


    def remove_punctuation_numbers_specialChar(text):
        punctuations_num = re.compile(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0-9]')  # numbers, special characters
        cleantext = punctuations_num.sub('', text)
        return cleantext
    

    def remove_stopwords(text, stopwords):  # remove stop words and join them back
        words = text.split()
        cleaned_words = [word for word in words if word not in stopwords] 
        return ' '.join(cleaned_words) 
    

    def calculate_frequency(train_data):    # count frequencies of class as well as words in each class
        class_frequency = {}  #store frequency of class
        word_frequency = {}  #store frequency of words
        for label, text in zip(train_data["labels"], train_data["objects"]):    
            if label not in class_frequency:
                class_frequency[label] = 0
                word_frequency[label] = {}

            class_frequency[label] += 1  # count frequency of each class
            cleaned_text = remove_punctuation_numbers_specialChar(text)
            words = cleaned_text.split()
            words = text.split()    # split each review into words 
            for word in words:
                word_frequency[label][word] = word_frequency[label].get(word, 0) + 1  # increment each word frequency for each class (default in .get is 0)
        return class_frequency, word_frequency
    
    stopwords = ["able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", 
                "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", 
                "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", 
                "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", 
                "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", 
                "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", 
                "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", 
                "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", 
                "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but",
                "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", 
                "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", 
                "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt",
                "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc",
                "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl",
                "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during",
                "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", 
                "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es",
                "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", 
                "ex", "exactly", "example", "except", "expect", "ey", "f", "f2", "fa", "far", "fc", "few","felt", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", 
                "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", 
                "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", 
                "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2",
                "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", 
                "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", 
                "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", 
                "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig",
                "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", 
                "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention",
                "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy",
                "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", 
                "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", 
                "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", 
                "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", 
                "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", 
                "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", 
                "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", 
                "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", 
                "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", 
                "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", 
                "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", 
                "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", 
                "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", 
                "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", 
                "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", 
                "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", 
                "relatively", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm",
                "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", 
                "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", 
                "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", 
                "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", 
                "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", 
                "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", 
                "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", 
                "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", 
                "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", 
                "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", 
                "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", 
                "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", 
                "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", 
                "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", 
                "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", 
                "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", 
                "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", 
                "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", 
                "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", 
                "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", 
                "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", 
                "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", 
                "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", 
                "yours", "yourself", "yourselves", "you've" ] 
    
    
    test_data["objects"] = [remove_stopwords(text, stopwords) for text in test_data["objects"]]
    test_data["objects"] = [remove_punctuation_numbers_specialChar(text) for text in test_data["objects"]]
    test_data["objects"] = [' '.join([stemmed_text(word) for word in text.split()]) for text in test_data["objects"]]

    class_frequency, word_frequency = calculate_frequency(train_data)
    prediction = [test_data["classes"][0]] * len(test_data["objects"])

    #calculate word probabilities using bayes net
    for i, doc in enumerate(test_data["objects"]):
        most_prob_class, highest_score = None, float("-inf")    #highest score will give you the class the word belongs to

        for label in test_data["classes"]:
            score = 0
            if label in class_frequency:
                score = math.log(class_frequency[label] / sum(class_frequency.values()))  #calculates the prior probability

                words = doc.split()    
                for word in words:  
                    if label in word_frequency and word in word_frequency[label]: #frequency of every word in test data
                        word_count = word_frequency[label][word]
                    else:
                        word_count = 0
                    score += math.log((word_count + 1) / (sum(word_frequency[label].values()) + len(word_frequency[label])))  #calculates the likelihood of the word
                    
                    # Laplace smoothing
                if score > highest_score:   #condition to check if the current class has the highest score
                    highest_score = score       
                    most_prob_class = label 

        prediction[i] = most_prob_class    # assigns the class with the highest score
    return prediction

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets from the command line arguments
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    
    # Check if the number of classes matches between training and test data and there are exactly 2 classes
    if (sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # Make a copy of the test data without the correct labels, so the classifier can't cheat
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    # Classify the test data using the Naive Bayes classifier
    results = classifier(train_data, test_data_sanitized)

    # Calculate and print the classification accuracy
    correct_ct = sum([(results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"]))])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
