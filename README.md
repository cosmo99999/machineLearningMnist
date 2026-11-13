# Machine learning implementation for mnist in C
- ml.c contains functions for matrix operations required to train a neural network
- including basic functions like matrix_mult and sigmoid function for activation
- it also contains larger functions like back_propagate and forward pass
- main.c just loads the training data and sets some parameters for training
- once a model has been trained, the parameters are saved to a binary file
- the executable can be passed no args to train a new model, or passed a filename of an existing model
- barebones usage of the curses lib to test the model with your own digits!
