import numpy as np
from scipy.special import expit, softmax, log_softmax
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, np.zeros(input.shape))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        zeros = np.zeros(input.shape)
        zeros[input > 0] = 1
        return grad_output * zeros


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        sigma = expit(input)
        return grad_output * sigma * (1 - sigma)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return softmax(input, axis=1)
    
    # https://themaverickmeerkat.com/2019-10-23-Softmax/
    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax1 = softmax(input, axis=1)
        dsoftmax = np.einsum('ij,jk->ijk', softmax1, np.eye(softmax1.shape[1])) - np.einsum('ij,ik->ijk', softmax1,
                                                                                            softmax1)
        return np.einsum('ij,ijk->ik', grad_output, dsoftmax)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return log_softmax(input, axis=1)

    # https://math.stackexchange.com/questions/2013050/log-of-softmax-function-derivative#:~:text=to%20me%2C%20the-,derivative,-of
    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax1 = softmax(input, axis=1)
        not_dsoftmax = np.einsum('ij,ik->ijk', np.ones(softmax1.shape) ,softmax1)
        return np.einsum("ij,ijk->ik", grad_output, np.eye(softmax1.shape[1]) - not_dsoftmax)