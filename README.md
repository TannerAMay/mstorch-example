# CS6406 HW1
## Missouri University of Science & Technology; Department of Computer Science


### Goals and directions:

- The main goal of this assignment is to implement perceptrons and neural networks from scratch and train them on any given dataset.
- Comprehend the impact of hyperparameters and learn to tune them effectively
- You are **not** allowed to use neural network libraries like PyTorch, Tensorflow, Keras, etc.
- You are also **not** allowed to add, move, or remove any files nor modify their names
- You are also **not** allowed to change function signatures
- You are also **not** allowed to modify the tests
- You *are* allowed to implement your code between the `# TODO: Replace below with your code >>>>>` and `# <<<<<` comments as well as add any functions you desire
- Please note that this code may take a while to run on a single CPU

### Problem 1 Neural Network Components *(5 points)*

- Implement a Linear Basis using the functions within `mstorch/nn/basis.py` file *(1 point)*
- Implement the ReLU and Sigmoid activations within `mstorch/nn/activation.py` file *(2 points)*
- Implement L2 loss within `mstorch/nn/loss.py` file *(2 points)*

### Problem 2 Models *(8 points)*

- Using the MSTorch library, implement the two layer neural network class `NN2` in `NN2.py`

    - Refer to `NN1.py` for an example implementation of a single layer network

### Problem 3 Optimization Algorithms *(6 points)*

- Implement the `zero_grad()` function in `mstorch/optim/optimizer.py` file *(2 points)*
- Implement the Random Coordinate Descent variant of SGD in `mstorch/optim/rcd.py` file *(2 points)*
- Implement the Adam optimizer in `mstorch/optim/adam.py` file *(2 points)*

### Problem 4 Classification on MNIST data *(6 points)*

- Implement the data preprocessing functions in `data_processing.py` *(2 points)*
- Implement the `train` and `test` function in `NN2.py` *(2 points)*
- Implement complete training, testing script in `if __name__ == "__main__"` portion of `NN2.py` *(2 points)*
