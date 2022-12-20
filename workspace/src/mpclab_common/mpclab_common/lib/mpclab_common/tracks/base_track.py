from abc import abstractmethod

class BaseTrack():
    @abstractmethod
    def global_to_local_typed(self,data):
        raise NotImplementedError('Cannot call base class')

    @abstractmethod
    def local_to_global_typed(self,data):
        raise NotImplementedError('Cannot call base class')

    @abstractmethod
    def get_curvature(self,s):
        raise NotImplementedError('Cannot call base class')

    @abstractmethod
    def get_halfwidth(self,s):
        raise NotImplementedError('Cannot call base class')

    def get_bankangle(self,s):
        raise NotImplementedError('Cannot call base class')
