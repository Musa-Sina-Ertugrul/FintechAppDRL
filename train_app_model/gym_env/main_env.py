import pygame as pg
import numpy as np
from itertools import repeat
import gymnasium as gym
from gym_env.data_structures import (
    DrawableNode,
    NodeType,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    ELLIPSE_HEIGHT,
    ELLIPSE_WIDTH,
    CENTER,
    CENTER_X,
    CENTER_Y,
    RADIUS,
)


class MainEnv:

    def __init__(self, surface: pg.Surface) -> None:
        self.__surface = surface

    def __draw_lines(self, points) -> None:
        for x, y in points:
            pg.draw.line(self.__surface, "black", start_pos=CENTER, end_pos=(x, y))

    def __draw_nodes(self, nodes, points_iter) -> None:
        for node in nodes:
            match node.color:
                case NodeType.CURRENT.value:
                    node.draw(
                        CENTER_X - ELLIPSE_WIDTH // 2, CENTER_Y - ELLIPSE_HEIGHT // 2
                    )
                case _:
                    x, y = next(points_iter)
                    node.draw(x - ELLIPSE_WIDTH // 2, y - ELLIPSE_HEIGHT // 2)
                    print(node.color, " ", x, " ", y)

    def draw(self, nodes: list[DrawableNode]) -> None:
        points = self._calculate_points(len(nodes) - 1)
        self.__draw_lines(iter(points))
        self.__draw_nodes(iter(nodes), iter(points))

    def __correct_coordinates_for_center(
        self, x: int, y: int, degree: float
    ) -> tuple[float]:
        return x - (ELLIPSE_WIDTH // 2) * np.cos(degree), y - (
            ELLIPSE_HEIGHT // 2
        ) * np.sin(degree)

    def _calculate_points(self, node_count: int) -> list[tuple[float, float]]:
        self.__check_node_count(node_count)
        step_degree = 360 / node_count
        x_values = self.__calculate_circle_x_values(node_count, step_degree)
        y_values = self.__calculate_circle_y_values(node_count, step_degree)
        packed_coordinates = []
        for x, y in zip(x_values, y_values):
            # x,y = self.__correct_coordinates_for_center(x,y,i*step_degree)
            packed_coordinates.append((x, y))
        return packed_coordinates

    def __check_node_count(self, node_count: int) -> None:
        if node_count < 1:
            raise ValueError(f"node_count: {node_count} is smaller than 1") from None

    def __calculate_circle_x_values(self, node_count: int, step_degree) -> list[float]:
        results = []
        for i in range(node_count):
            results.append(CENTER_X - RADIUS * np.cos(i * step_degree))
        return results

    def __calculate_circle_y_values(self, node_count: int, step_degree) -> list[float]:
        results = []
        for i in range(node_count):
            results.append(CENTER_Y - RADIUS * np.sin(i * step_degree))
        return results
