# MSTorch ⛏️🔥
### Deep-Learning Education the *right* way with a faithful simplification of PyTorch
#### Written for *M*issouri *S*&*T* CS6406 by Tanner May

Dependencies: Python 3.10 and Numpy. (and PyTorch for unit tests)  
This project uses [pipenv](https://github.com/pypa/pipenv) for dependencies. Simply insure you have pipenv and Python 3.10 installed then run `pipenv install` from the root of the repository.

The goal of this project is to learn deep learning the *right* way. That is, implementing the algorithms from scratch while mimicking an existing architecture. PyTorch's architecture is easy to understand and iterate upon due its extensive use of object-oriented programming and the fact that essential DL processes aren't abstracted away, like training. By using mimicking this architecture, students will leave the course with an understanding of the algorithms, DL pipelines from data retrieval to model output, and PyTorch. Our goal is like that of CS1575: teach students the algorithms while simultaneously teaching them how to use libraries they will use professionally.

The library was also built with [grade.sh](https://gitlab.com/classroomcode/grade-sh) in mind, hence it has (some) unit tests implemented for its basic functionality. The unit tests randomly generate inputs and compare MSTorch outputs to PyTorch, thus the requirement for `torch` when running test files, not only for forward processing but also gradient calculations. Since the inner workings of PyTorch can be complicated, not all MSTorch features are tested. This area needs future work, ideally before deploying as an assignment. By finishing these unit tests, grading for the whole assignment can be completely automated.

To prepare the library for an assignment, simply remove the code in between the `# TODO: Replace below with your code >>>>>` and `# <<<<<` comments. For example, the code below:
```python
# TODO: Replace below with your code >>>>>
# print("No answers here!")

print("This is the answer!")

# <<<<<
```
Becomes:
```python
# TODO: Replace below with your code >>>>>
print("No answers here!")

# <<<<<
```
Also be sure to check that the commit history does not allow for answer recovery. The `if __name__ == "__main__"` portion of each file should remain intact, it is meant to show students how the file will be used.

Regarding commits, this repository uses the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) standard, you should too.

There may be concerns regarding cheating since this library is similar to PyTorch, ie copying PyTorch's code. Let me assure you, that, after reading its source code, PyTorch's code needs significant modifications to work on MSTorch, not to mention the effort required to understand PyTorch's source. PyTorch has *many* more features than MSTorch that are tightly integrated, this makes it difficult to understand what part of the source is actually relevant. Despite this, some students may turn to looking at PyTorch's source for some inspiration, but is this really so bad? The students are reading the source code for a real library that they will actually use in their endeavors, not many other courses can say the same.


## Folder Structure

- The `nn` directory: Contains the basic building blocks of networks

    - `activation.py`: `class` implementations of activation functions' forward and backward methods.
    - `basis.py`: `class` implementations of basis functions' forward and backward methods.
    - `loss.py`: `class` implementations of loss functions' forward and backward methods.
    - `module.py`: Analogous to PyTorch's `Module`. This implementation is extremely bare bones and not essential to operation of MSTorch; it was included so students are aware of its prevalence in PyTorch.

- The `optim` directory: Contains optimization functions

    - `adam.py`: `class` implementation of Adam optimizer.
    - `optimizer.py`: Analogous to PyTorch's `Optimizer`. Included for the same reasons as `nn.Module`
    - `rcd.py`: `class` implementation of Random Coordinate Descent optimizer discussed in lecture.

The library can be easily extended by simply adding `class` implementations of the desired algorithms, just like PyTorch!
