Hangman Solver
A Deep Learning based approach to solving Hangman Game 


# Setup Environment 
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
# Project Structure
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
# Running the hangman solver
