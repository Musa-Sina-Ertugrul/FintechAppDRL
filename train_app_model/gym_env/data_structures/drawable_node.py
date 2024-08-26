import pygame as pg
from .node import Node
from .node_type_n_consts import ELLIPSE_HEIGHT, ELLIPSE_WIDTH, NodeType


class DrawableNode(Node):

    font: pg.font.Font = pg.font.SysFont(pg.font.get_fonts()[0], 20)

    def __init__(
        self, name: str, number: int, node_type: NodeType, id: int, surface: pg.Surface
    ) -> None:
        super().__init__(name, number, node_type, id)
        self.__surface = surface
        self.__pygame_text = self.font.render(self.name, False, "black", "white")
        self.__text_x, self.__text_y = self.__pygame_text.get_size()

    def __bool__(self):
        return super().__bool__()

    def draw(self, x: int, y: int) -> None:
        center_x, center_y = x + ELLIPSE_WIDTH // 2, y + ELLIPSE_HEIGHT // 2
        rect = pg.Rect(x, y, ELLIPSE_WIDTH, ELLIPSE_HEIGHT)
        ellipse: pg.Rect = pg.draw.ellipse(self.__surface, self.color, rect)
        self.__surface.blit(
            self.__pygame_text,
            (center_x - self.__text_x // 2, center_y - self.__text_y // 2),
        )
