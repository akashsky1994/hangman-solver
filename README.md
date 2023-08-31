# Hangman Solver
A Deep Learning based approach to solving Hangman Game 


## Setup Environment 
This project uses poetry for environment management
```
poetry install
```

If you are using conda or pip just install the given dependencies
```
Pandas
Numpy
Torch - if using model inference
tqdm
```
## Project Structure
```
├── README.md
├── checkpoints
│   ├── model_4.581_0.0
│   │   ├── adam.pth
│   │   ├── model.pth
│   │   └── model_state_dict.pth
├── hangman
│   ├── __init__.py
│   ├── dataset.py
│   ├── hangman_api_user.ipynb
│   ├── model.py
│   ├── ngram.py
│   ├── strategy.py
│   ├── trainer.py
│   └── words_250000_train.txt
├── poetry.lock
├── pyproject.toml
├── test.txt
└── tests
```
## Running the hangman solver

We use ```hangman_api_user.ipynb``` as our running notebook. Different strategies have been implemented which you can find in the file ```strategy.py```. Default one is currently the bidirectional ngram character model. 

## Intuition behind each strategies:


### Information theory - Calculating Entropy (Initial Theory) 
Information theory tells us, that the “information” of an event is -log2(p) where p is the probability of the event (the choice of a base 2 log is arbitrary but makes the calculations a bit nicer). This means a more probable event contains more information than a less probable one. Lastly we have entropy which is simply the expected value (or weighted sum) of the information of an event. It tells us the amount of information, on average, that we will receive from a set of events which in our case would be picking any letter. 
So we aim to calculate entropy for all letters of alphabet for our available dictionary (dictionary after filtered to the current obscured word and length), whichever has the highest entropy is picked as our next guess. 
Refer to class ```EntropyIT``` in ```strategy.py``` for implementation.

### Character Ngram Model (Best performing)
Dataset Preparation:

1.	After looking at the dictionary dataset, I found a lot of junk words. I used a simple filtering strategy to remove words which have only 1 set of characters like “aaaaaa”, “zzzz”, …. This reduced my training set from 250,000 to 227,259.

2.	Usually, the ngram text generation models work with a pair: (prefix context tokens, target token). Since the game revolves around guessing characters not necessarily in the left-to-right approach, I added pairs of (suffix context tokens, target token)

3.	For each word in the dictionary, I generated such training pairs. For example, “hello” would have these pairs for my character level trigram model.

Prefix Context Ngram pairs:
((<start>, <start>), h)
((<start>, h), e)
((h, e), l)
((e, l), l)
((l, l), o)

Suffix Context Ngram pairs:
((<end>, <end>), o)
((o, <end>), l)
((l, o), l)
((l, l), e)
((e, l), h)

4.	Across all the words, I calculate the occurences of each:
a.	Set of : Prefix context tokens ----- n1
b.	Set of : Suffix context tokens ----- n2
c.	Pair of : (Prefix context tokens, target token) ----- n3
d.	Pair of : (Suffix context tokens, target token) ----- n4


Character Ngram Model:

1.	My model approach is simply calculating probability of a token given its prefix context tokens and suffix context tokens.

2.	In other words,

P1 = Number of Pairs of (Prefix context tokens, target token) / Number of sets of Prefix context tokens
Or, P1 = n1/n3

P2 = Number of Pairs of (Suffix context tokens, target token) / Number of sets of Suffix context tokens
Or, P2 = n2/n4

3.	For the final probability, I take max of P1 and P2 calculated to determine the final probabilities for each character token.

P = max(P1, P2)

I tried using averaging, however, found that max operation yielded better results.

Vowel Model:

1.	In my several games of hangman, I found that the best strategy is to start by guessing vowels depending on the length of the word.

2.	To implement this, I simply grouped all words in the dictionary by their length first. Next, I calculated these for each unique word length:
a.	Average number of vowels
b.	Average individual probabilities of occurrence of [“a”, “e”, “i”, “o”, “u”]


### Bi-directional LSTM Model Prediction (Additional Strategy Using Deep learning model Model file not available at the time of submission) 
We use a bidirectional LSTM model to train our hangman solver. We convert each character into an embedding 1x26 where each index indicates a letter of alphabet. Consequently, we get a tensor nx26 for each word where n is the length of word. This input along with the guessed character list is added to model for inference, loss calculated and further optimization. We concatenate the guessed letters to the output of the lstm and pass it to a linear layer which has an outputs of shape nx26. 
Refer to ```trainer.py``` for implementation. 

I am looking at MLM (Masked Language Model) with Character BERT which might give better results but due to time constraint, I have not been able try out that strategy.
