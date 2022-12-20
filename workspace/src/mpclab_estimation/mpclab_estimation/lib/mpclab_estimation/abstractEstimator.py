#!/usr/bin python3

from abc import ABC, abstractmethod

class abstractEstimator(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def update(self):
        pass
