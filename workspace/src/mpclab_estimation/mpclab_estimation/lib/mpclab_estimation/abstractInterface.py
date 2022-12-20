#!/usr/bin python3

from abc import ABC, abstractmethod

'''
The interface class is meant to preprocess the raw measurements recieved from
the sensor and to transform the readings into the correct frame and to
account for any offsets there may be between the CoM of the vehicle and sensor placement

e.g. x, y, z in gps frame -> x, y, z in track frame

In the abstract class, the get_com_meas method is required in all implementations.
'''
class abstractInterface(ABC):

    @abstractmethod
    def get_com_meas(self):
        pass
