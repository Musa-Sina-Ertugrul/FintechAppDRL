
class Node:
    
    def __init__(self,name:str,number:int,is_wrong:bool,id:int) -> None:
        self.__name = name
        self.__id = id
        self.__children = []
        self.__number = number
        self.__is_wrong = is_wrong
        self.__parent : 'Node'= None
    
    @property
    def name(self):
        return self.__name
    
    def __str__(self):
        return self.name
    
    @property
    def parent_ptr(self):
        return self.__parent
    
    @property
    def parent(self):
        return str(self.__parent)

    def add_child(self,node:'Node')->None:
        self.__children.append(node)
    
    def __int__(self):
        return self.__number

    @property
    def id(self):
        return self.__id
    
    @property
    def __bool__(self):
        return self.__is_wrong