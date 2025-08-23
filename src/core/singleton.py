"""Singleton metaclass and decorator implementation."""
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")

class Singleton(type):
    """Singleton metaclass for ensuring only one instance of a class exists."""
    
    _instances: Dict[Type[Any], Any] = {}
    
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Call method for the singleton metaclass.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            Any: The singleton instance of the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

def singleton(cls: Type[T]) -> Type[T]:
    """Singleton decorator for classes.
    
    Args:
        cls: The class to be decorated.
        
    Returns:
        Type[T]: The decorated class with singleton behavior.
    """
    instances: Dict[Type[T], T] = {}
    
    def get_instance(*args: Any, **kwargs: Any) -> T:
        """Get the singleton instance of the class.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            T: The singleton instance.
        """
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance
