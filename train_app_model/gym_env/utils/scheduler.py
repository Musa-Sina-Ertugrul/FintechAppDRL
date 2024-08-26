from typing import Callable
from types import MethodType


class Scheduler:

    def __init__(self, value: float, decay_func: Callable):
        self.__value = value
        self.__current_epoch = 0
        self.__decay_func = MethodType(decay_func, self)

    @property
    def value(self) -> float:
        return self.__value

    @property
    def current_epoch(self) -> int:
        return self.__current_epoch

    def step(self) -> None:
        self.__value = self.__decay_func()
        self.__current_epoch += 1


def linear_decay(end_iter: int) -> Callable:
    last_epoch = end_iter

    def inner_linear_decay(self: Scheduler) -> None:
        if last_epoch < self.current_epoch:
            raise RuntimeError(
                f"last epoch: {last_epoch}, current epoch {self.current_epoch}.\
                end_iter passed line. Raise end_iter"
            ) from None
        return self.value - self.current_epoch * self.value / last_epoch

    return inner_linear_decay
