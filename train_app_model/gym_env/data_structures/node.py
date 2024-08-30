from enum import EnumType
from typing import Union


class Node:

    def __init__(
        self, name: str, number: int, node_type: EnumType, id: int, is_finish: bool
    ) -> None:
        self.__name = name
        self.__id = id
        self.__children = {}
        self.__number = number
        self.__node_type = node_type
        self.__is_finish = is_finish

    @property
    def name(self):
        return self.__name

    def __str__(self):
        return self.name

    @property
    def reward(self):
        return self.__number

    @property
    def parent(self):
        return str(self.__parent)

    def is_child(self, node: Union["Node", str]) -> bool:
        children_set = set(list(self.__children.keys()))
        try:
            return node.name in children_set
        except AttributeError:
            return str(node) in children_set
        except BaseException as e:
            raise e

    def add_child(self, node: "Node") -> "Node":
        if self.__children.get(node.name, None) is None:
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

    def __bool__(self):
        return self.__is_finish
