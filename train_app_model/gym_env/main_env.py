import pygame as pg
import gymnasium as gym
import numpy as np
import pandas as pd
from itertools import count
from gym_env.data_structures import Node, NodeType, DrawableNode
from main_env_pygame import MainEnv


class FintechAppDRLEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, memory_len: int):
        super().__init__()
        pg.init()
        self.__screen = pg.display.set_mode((1280, 720))
        self.__clock = pg.time.Clock()
        self.__memory_len = memory_len
        self.__env_map = self.__create_agent_map()
        self.__current_node: Node = None
        self.__past_states = []

    def __create_agent_map(self) -> dict[Node]:
        df = pd.read_csv("/home/musasina/projects/FintechAppDRL/dataset/appdata10.csv")
        df = df.drop("user", axis=1)
        df = df.drop("first_open", axis=1)
        df = df.drop("dayofweek", axis=1)
        df = df.drop("hour", axis=1)
        df = df.drop("age", axis=1)
        df = df.drop("minigame", axis=1)
        df = df.drop("used_premium_feature", axis=1)
        df = df.drop("enrolled", axis=1)
        df = df.drop("enrolled_date", axis=1)
        df["numscreens"] = df["numscreens"].astype(int)
        df["liked"] = df["liked"].astype(int)
        df = df[df["numscreens"] == 65]
        df_liked = df[df["liked"] == 1]
        df_not_liked = df[df["liked"] == 0]
        screens_liked = df_liked["screen_list"].to_list()
        screens_not_liked = df_not_liked["screen_list"].to_list()
        separated_screens_liked = []
        for user_screens in screens_liked:
            separated_screens_liked.append(user_screens.split(","))
        separated_screens_not_liked = []
        for user_screens in screens_not_liked:
            separated_screens_not_liked.append(user_screens.split(","))
        unique_screens = []

        for separated_user_screens in (
            separated_screens_liked + separated_screens_not_liked
        ):
            unique_screens.extend(np.unique(separated_user_screens))
        unique_screens_number_map = {}
        number_unique_screen_map = {}
        for i, screen in zip(count(1), unique_screens):
            unique_screens_number_map[screen] = i
            number_unique_screen_map[i] = screen
        self.__number_unique_screen_map = number_unique_screen_map
        self.__unique_screens_number_map = unique_screens_number_map
        env_map = {}
        node_count = 0
        for user_screens in separated_screens_liked:
            node_ptr = None
            for i, screen in enumerate(user_screens):
                if i == 0 and hasattr(env_map, screen):
                    node_ptr = Node(
                        screen,
                        unique_screens_number_map[screen],
                        NodeType.TRUE,
                        node_count,
                    )
                    env_map[screen].add_child(node_ptr)
                elif i == 0 and not hasattr(env_map, screen):
                    node_ptr = Node(
                        screen,
                        unique_screens_number_map[screen],
                        NodeType.TRUE,
                        node_count,
                    )
                    env_map[screen] = node_ptr
                else:
                    new_node = Node(
                        screen,
                        unique_screens_number_map[screen],
                        NodeType.TRUE,
                        node_count,
                    )
                    node_ptr = node_ptr.add_child(new_node)
                node_count += 1

        separated_screens_not_liked = []

        for user_screens in separated_screens_not_liked:
            node_ptr = None
            for i, screen in enumerate(user_screens):
                if i == 0 and hasattr(env_map, screen):
                    node_ptr = env_map[screen]
                elif i == 0 and not hasattr(env_map, screen):
                    break
                else:
                    new_node = Node(
                        screen,
                        unique_screens_number_map[screen],
                        NodeType.WRONG,
                        node_count,
                    )
                    node_ptr = node_ptr.add_child(new_node)
        return env_map

    def _push(self, node_name: str) -> None:
        self.__past_states.insert(0, self.__unique_screens_number_map[node_name])
        if len(self.__past_states) > self.__memory_len:
            self._pop()

    def _pop(self) -> int:
        return self.__past_states.pop()

    def step(self, action: int) -> tuple[list[int], int, int, bool]:
        self.render()
        if self.__current_node.is_child(self.__number_unique_screen_map[action]):
            retrieved_node = self.__current_node.return_child(
                self.__number_unique_screen_map[action]
            )
            if retrieved_node is None:
                checked_value = NodeType.check_wrong_true(self.__current_node)
                return self.__past_states, checked_value, checked_value, True
            elif self.__current_node.has_child():
                checked_value = NodeType.check_wrong_true(retrieved_node)
                self._push(retrieved_node)
                self.__current_node = retrieved_node
                return self.__past_states, checked_value, checked_value, False
            else:
                raise RuntimeError(
                    f"There is no child like that check child {retrieved_node.name}"
                )
        return self.__past_states, -1, -1, False

    def reset(self, seed=None, options=None) -> list[int]:
        map_start = np.random.choice(list(self.__env_map.keys()))
        self.__past_states = [0] * (self.__memory_len - 1)
        self._push(map_start)
        return self.__past_states

    def render(self):
        self.__screen.fill("white")
        main_pygame_env = MainEnv(self.__screen)
        nodes = [node for node in self.__current_node.return_children()] + [
            self.__current_node
        ]
        drawable_nodes = [
            DrawableNode(
                node.name,
                -1,
                NodeType.retrieve_node_type(node.color, -1, self.__screen),
            )
            for node in nodes
        ]
        
        main_pygame_env.draw(drawable_nodes)
        pg.display.flip()
        self.__clock.tick(60)

    def close(self):
        pg.quit()
        del self
