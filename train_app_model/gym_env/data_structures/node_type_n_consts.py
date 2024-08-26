from enum import Enum

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
ELLIPSE_HEIGHT = 100
ELLIPSE_WIDTH = 200
CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
CENTER_X, CENTER_Y = CENTER

PAD_X = ELLIPSE_WIDTH // 2
PAD_Y = ELLIPSE_HEIGHT // 2

RADIUS = 260


class NodeType(Enum):
    CURRENT = "purple"
    WRONG = "red"
    TRUE = "green"

    @classmethod
    def retrieve_node_type(cls, node_color: str):
        match node_color:
            case "purple":
                return cls.CURRENT
            case "red":
                return cls.WRONG
            case "green":
                return cls.TRUE
    @classmethod
    def check_wrong_true(cls, node) -> bool:
        match node.color:
            case "red":
                return -1
            case "green":
                return 1
            case _:
                raise RuntimeError(f"Wrong color has come {node.color}")


class CartesianCoordinateSections(Enum):
    SECTION_1 = (1, 1)
    SECTION_2 = (-1, 1)
    SECTION_3 = (-1, -1)
    SECTION_4 = (1, -1)
