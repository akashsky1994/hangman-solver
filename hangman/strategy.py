import pandas as pd
import math
import collections
import re
from trainer import Trainer
from ngram import CharacterNGramBiDirectionalModel

########################################
########## Base Strategy Class ##########
########################################
class BaseStrategy:
    def __init__(self, full_dictionary):
        self.full_dictionary = full_dictionary
        self.current_dictionary = []
        self.ALPHABET = {"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"}

    def generate_optimal_choice(self, obscured_word, guessed_letters):
        raise NotImplementedError()
    
    def get_current_dictionary(self,word):
        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_",".")
        
        # find length of passed word
        len_word = len(clean_word)
        
        new_dictionary = []
        
        # iterate through all of the words in the old plausible dictionary
        for dict_word in self.full_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        return self.current_dictionary

########################################
###### Base Algorithm (Most Freq) ######
########################################
class MostFrequent(BaseStrategy):
    def __init__(self, full_dictionary):
        super().__init__(full_dictionary)
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()

    def generate_optimal_choice(self, obscured_word, guessed_letters):
        # Generate updated dictionary
        dictionary = self.get_current_dictionary(obscured_word)

        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(dictionary)
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in guessed_letters:
                    guess_letter = letter
                    break  
        return guess_letter          
        
########################################
########## Information theory ##########
########################################
class EntropyIT(BaseStrategy):
    def __init__(self, full_dictionary):
        super().__init__(full_dictionary)
        

    def generate_optimal_choice(self, obscured_word, guessed_letters):
        # Generate updated dictionary
        dictionary = self.get_current_dictionary(obscured_word)
        
        # Creating a dataframe
        dictionary_df = pd.DataFrame({'word':dictionary})
        available_letters = list(self.ALPHABET -set(guessed_letters))
        entropy_lookup = {}

        # Get entropy for all available letters
        for letter in available_letters:
            entropy_lookup[letter] = self.compute_entropy(dictionary_df,letter)
        
        # print(entropy_lookup)
        # return the letter with highest entropy
        guess = max(entropy_lookup, key=entropy_lookup.get)
        return guess

    def compute_entropy(self, current_dictionary_df, letter):
        total_count = current_dictionary_df.shape[0]
        ##################
        # Generate All Possible Pattern that can be generated from the given letter 
        ##################
        current_dictionary_df['pattern'] = current_dictionary_df['word'].apply(lambda word: self.get_pattern(word, letter))
        current_dictionary_df = current_dictionary_df[current_dictionary_df['pattern'].isnull() == False]  
        
        ##################
        # Group by patterns and save count into the dataframe
        ##################
        current_dictionary_df = current_dictionary_df.value_counts(subset=["pattern"]).reset_index(name="count")
        
        #################
        # Calculating Entropy sum(-p.log2(p)) where p is probability of a certain pattern in the available set of words/dictionary
        #################
        current_dictionary_df['entropy'] = current_dictionary_df['count'].apply(lambda x: -math.log(x/total_count)*(x/total_count))
        
        return current_dictionary_df['entropy'].sum()
        
    def get_pattern(self, word, letter):
        pattern = ''
        if letter not in word:
            return None
        for c in word:
            if c==letter:
                pattern += "1"
            else:
                pattern += "0"
        return pattern

######################################################
########### Character Ngram Bidirectional  ###########
######################################################
class NGramPrediction(BaseStrategy):
    def __init__(self, full_dictionary):
        super().__init__(full_dictionary)
        self.get_current_dictionary()
        
        self.build_vowel_probabilities(vowels=["a", "e", "i", "o", "u"])
        self.models = [
            CharacterNGramBiDirectionalModel(2, self.current_dictionary),
            CharacterNGramBiDirectionalModel(3, self.current_dictionary),
            CharacterNGramBiDirectionalModel(4, self.current_dictionary),
            CharacterNGramBiDirectionalModel(5, self.current_dictionary),
            CharacterNGramBiDirectionalModel(6, self.current_dictionary),
        ]
        # self.weights = [0.05, 0.2, 0.3, 0.3, 0.15]
        self.weights = [1, 1, 1, 1, 1]

    def get_current_dictionary(self):
        self.current_dictionary = [word for word in self.full_dictionary if len(set(word)) != 1]
        
    def generate_optimal_choice(self, obscured_word, guessed_letters):
        clean_word = obscured_word[::2].replace("_",".")
        
        # find length of passed word
        len_word = len(clean_word)
        
        if len_word == clean_word.count("."):
            guess_letter = self.next_vowel_predictor(clean_word, len_word, guessed_letters)
        else:
            guess_letter = self.guess_next_ch(clean_word, len_word, guessed_letters)
            
        return guess_letter

    def next_vowel_predictor(self, word, key, exclude_list):
        while True:
            if key in self.num_vowel_probabilities:
                num_vowel = self.num_vowel_probabilities[key]
                break
            else:
                key -= 1
        vowel_probabilities = self.guess_vowel_probabilities[key]
        vowel_probabilities = sorted(list(vowel_probabilities.items()), key=lambda x: x[1], reverse=True)
        vowel = None
        while len(vowel_probabilities):
            vowel, prob = vowel_probabilities.pop(0)
            if vowel not in exclude_list:
                return vowel
        return vowel
    
    def guess_next_ch(self, word, key, exclude_list):
        next_ch = []
        for i in range(len(word)):
            if word[i] != ".":
                continue
            prev = word[:i]
            post = word[i+1:]
            probabilities = self.calculate_for_contexts(prev, post)
            while True:
                ch, prob = probabilities.pop(0)
                if ch not in exclude_list:
                    next_ch.append((i, ch, prob))
                    break
        next_ch.sort(key=lambda x: x[2], reverse=True)
        return next_ch[0][1]
    
    def calculate_for_contexts(self, prev_context, post_context):
        return self.combine_probabilities([
            model.calculate_probability(prev_context, post_context)
            for model in self.models
        ])
    
    def combine_probabilities(self, probabilities):
        ch_set = set()
        num = len(probabilities)
        for prob in probabilities:
            ch_set = ch_set.union(prob.keys())
        final_probablity = []
        for ch in ch_set:
            sum_prob_list = max([prob.get(ch, 0.0)*weight for prob, weight in zip(probabilities, self.weights)])
            final_probablity.append((ch, sum_prob_list))
        return sorted(final_probablity, key=lambda x: x[1], reverse=True)

    def build_vowel_probabilities(self, vowels):
        self.guess_vowel_probabilities = {}
        self.num_vowel_probabilities = {}
        len_wise_words = {}
        for word in self.current_dictionary:
            key = len(word)
            total_vowel_count = 0
            current_dict = self.guess_vowel_probabilities.get(key, dict().fromkeys(vowels, 0))
            current_num_dict = self.num_vowel_probabilities.get(key, [])
            for ch in vowels:
                if ch in word:
                    current_dict[ch] += 1
                total_vowel_count += word.count(ch)
            current_num_dict.append(total_vowel_count)    
            self.guess_vowel_probabilities[key] = current_dict
            self.num_vowel_probabilities[key] = current_num_dict
            len_wise_words[key] = len_wise_words.get(key, 0) + 1
        for key in self.guess_vowel_probabilities:
            self.num_vowel_probabilities[key] = sum(self.num_vowel_probabilities[key])/len_wise_words[key]
            for ch in vowels:
                self.guess_vowel_probabilities[key][ch] = self.guess_vowel_probabilities[key][ch]/len_wise_words[key]
     
########################################
########### Model Prediction ###########
########################################
class ModelInference(BaseStrategy):
    def __init__(self, full_dictionary):
        super().__init__(full_dictionary)
        self.model = Trainer(infer=True)

    def generate_optimal_choice(self, obscured_word, guessed_letters):
        clean_word = obscured_word.strip().replace(" ","")
        return self.model.infer(clean_word, guessed_letters)



def get_strategy(strategy_mode) -> BaseStrategy:
    if strategy_mode == "most_freq":
        strategy = MostFrequent
    elif strategy_mode == "entropy_it":
        strategy = EntropyIT
    elif strategy_mode == "ngram":
        strategy = NGramPrediction
    elif strategy_mode == "lstm":
        strategy = ModelInference
    else:
        strategy = MostFrequent

    return strategy
