import pygame as pg
import gymnasium as gym
import numpy as np
import pandas as pd
from itertools import count
from gym_env.data_structures import Node, NodeType, DrawableNode
from .main_env_pygame import MainEnv
from sklearn.feature_extraction.text import CountVectorizer


class FintechAppDRLEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    font: pg.font.Font = pg.font.SysFont(pg.font.get_fonts()[0], 44)

    def __init__(self, memory_len: int):
        super().__init__()
        pg.init()
        self.__true_finished_count = 0
        self.__screen = pg.display.set_mode((1280, 720))
        self.__clock = pg.time.Clock()
        self.__memory_len = memory_len
        self._env_map = self.__create_agent_map()
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
        df = df[df["numscreens"] > 10]
        df = df[df["numscreens"] < 50]
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
        unique_screens = list(CountVectorizer().fit(unique_screens).vocabulary_)
        for i, screen in zip(count(2), unique_screens):
            unique_screens_number_map[screen] = i 
            number_unique_screen_map[i] = screen
        unique_screens_number_map["pad_screen"] = 0
        number_unique_screen_map[0] = "pad_screen"
        unique_screens_number_map["custom_screen"] = 1
        number_unique_screen_map[1] = "custom_screen"
        self.__number_unique_screen_map = number_unique_screen_map
        self.__unique_screens_number_map = unique_screens_number_map
        keys_set = set(unique_screens)
        for i,user_screens in enumerate(separated_screens_liked.copy()):
            for j,screen in enumerate(user_screens):
                if screen not in keys_set:
                    separated_screens_liked[i][j] = "custom_screen"
        for i,user_screens in enumerate(separated_screens_not_liked.copy()):
            for j,screen in enumerate(user_screens):
                if screen not in keys_set:
                    separated_screens_not_liked[i][j] = "custom_screen"
        results = []
        for user_screens in separated_screens_liked:
            past_screen = ""
            squeezed_screens = []
            for screen in user_screens:
                if past_screen != screen:
                    squeezed_screens.append(screen)
                    past_screen = screen
            results.append(squeezed_screens)
        separated_screens_liked = results
        for user_screens in separated_screens_not_liked:
            past_screen = ""
            squeezed_screens = []
            for screen in user_screens:
                if past_screen != screen:
                    squeezed_screens.append(screen)
                    past_screen = screen
            results.append(squeezed_screens)
        separated_screens_not_liked = results
        env_map = {}
        for user_screens in separated_screens_liked:
            node_ptr = None
            past_node_ptr = None
            for i, screen in enumerate(user_screens):
                if i == 0 and hasattr(env_map, screen):
                    node_ptr = env_map[screen]
                elif i == 0 and not hasattr(env_map, screen):
                    node_ptr = Node(
                        screen,
                        -(len(user_screens)-i)/len(user_screens),
                        NodeType.TRUE,
                        unique_screens_number_map[screen],i ==(len(user_screens)-1)
                    )
                    env_map[screen] = node_ptr
                else:
                    new_node = Node(
                        screen,
                        -(len(user_screens)-i)/len(user_screens),
                        NodeType.TRUE,
                        unique_screens_number_map[screen],i ==(len(user_screens)-1)
                    )
                    past_node_ptr = node_ptr
                    node_ptr = node_ptr.add_child(new_node)
                    node_ptr.add_child(past_node_ptr)

        for user_screens in separated_screens_not_liked:
            past_node_ptr = None
            node_ptr = None
            for i, screen in enumerate(user_screens):
                if i == 0 and hasattr(env_map, screen):
                    node_ptr = env_map[screen]
                elif i == 0 and not hasattr(env_map, screen):
                    node_name = np.random.choice(list(env_map.keys()))
                    node_ptr = env_map[node_name]
                else:                        
                    new_node = Node(
                        screen,
                        -1,
                        NodeType.WRONG,
                        unique_screens_number_map[screen],i ==(len(user_screens)-1)
                    )
                    past_node_ptr = node_ptr
                    node_ptr = node_ptr.add_child(new_node)
                    node_ptr.add_child(past_node_ptr)
        return env_map

    def _push(self, node_name: str) -> None:
        self.__past_states.insert(0, self.__unique_screens_number_map[node_name])
        if len(self.__past_states) > self.__memory_len:
            self._pop()

    def _pop(self) -> int:
        return self.__past_states.pop()

    def step(self, action: int) -> tuple[list[int], int, bool]:
        self.render()
        if self.__current_node.is_child(self.__number_unique_screen_map[action]):
            retrieved_node = self.__current_node.return_child(
                self.__number_unique_screen_map[action]
            )
            if self.__current_node.has_child():
                self._push(retrieved_node.name)
                self.__current_node = retrieved_node
            elif self.__current_node is None:
                raise RuntimeError(
                    f"There is no child like that check child {retrieved_node.name}"
                )
            if bool(self.__current_node):
                print(f"finished with {NodeType.retrieve_bool_text(self.__current_node)}")
                print(f"true finished count: {self.__true_finished_count}")
                print("-"*20)
                self.__true_finished_count += 1
                return self.__past_states, NodeType.check_wrong_true(self.__current_node), True
            return self.__past_states, self.__current_node.reward, bool(self.__current_node)
        return self.__past_states, -2, bool(self.__current_node)

    def reset(self, seed=None, options=None) -> list[int]:
        self.__past_states = [0] * (self.__memory_len - 1)
        if seed is None:
            map_start : str = np.random.choice(list(self._env_map.keys()))
        else:
            keys_listed = list(self._env_map.keys())
            if len(keys_listed) <= seed:
                raise RuntimeError(f"possible seed 0 to {len(keys_listed)-1}, yours seed is {seed}")
            map_start = keys_listed[seed]
        self._push(map_start)
        self.__current_node = self._env_map[map_start]
        self.render()
        return self.__past_states

    def render(self):
        self.__screen.fill("white")
        main_pygame_env = MainEnv(self.__screen)
        nodes = self.__current_node.return_children()
        drawable_nodes = [
            DrawableNode(
                node.name,
                -1,
                NodeType.retrieve_node_type(node.color), -1, self.__screen
            )
            for node in nodes
        ] + [ DrawableNode(self.__current_node.name,-1,NodeType.CURRENT,-1,self.__screen)]
        rendered_text = self.font.render(str(NodeType.retrieve_bool_text(self.__current_node)),False,self.__current_node.color,"white")
        main_pygame_env.draw(drawable_nodes)
        self.__screen.blit(rendered_text,(50,50))
        pg.display.flip()
        self.__clock.tick(60)

    def close(self):
        pg.quit()
        del self
