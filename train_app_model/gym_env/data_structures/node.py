from enum import EnumType


class Node:

    def __init__(self, name: str, number: int, node_type: EnumType, id: int) -> None:
        self.__name = name
        self.__id = id
        self.__children = {}
        self.__number = number
        self.__node_type = node_type
        self.__parent: "Node" = None

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

    def is_child(self, node: "Node") -> bool:
        children_set = set(list(self.__children.keys()))
        return node.name in children_set

    def add_child(self, node: "Node") -> "Node":
        if not hasattr(self.__children, node.name):
            self.__children[node.name] = node
        return self.__children[node.name]

    def return_child(self, name) -> "Node":
        return self.__children.get(name, None)

    def return_children(self) -> list["Node"]:
        return list(self.__children.values())

    def has_child(self) -> bool:
        return bool(self.__children)

    def __int__(self):
        return self.__number

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def id(self):
        return self.__id

    @property
    def color(self):
        return self.__node_type.value
