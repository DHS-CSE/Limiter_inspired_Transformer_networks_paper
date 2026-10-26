from abc import ABC, abstractmethod

class BaseAdapter(ABC):

    @abstractmethod
    def get_domain_coords(self, data):
        pass

    @abstractmethod
    def get_left_boundary_coords(self, data):
        pass

    @abstractmethod
    def get_right_boundary_coords(self, data):
        pass

    @abstractmethod
    def get_initial_coords(self, data):
        pass

    @abstractmethod
    def forward(self, model, coords, data, sharp_const=None):
        pass
