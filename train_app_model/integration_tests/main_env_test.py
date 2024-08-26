import os

os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame as pg

pg.init()
import sys

sys.path.append("/home/musasina/projects/FintechAppDRL/train_app_model")
from gym_env.data_structures import DrawableNode
from gym_env import MainEnv
from gym_env.data_structures.node_type_n_consts import NodeType

screen = pg.display.set_mode((1280, 720))
clock = pg.time.Clock()

node_true = DrawableNode("test1", -1, NodeType.TRUE, -1, screen)
node_wrong = DrawableNode("test2", -1, NodeType.WRONG, -1, screen)
node_current = DrawableNode("test3", -1, NodeType.CURRENT, -1, screen)
main_env = MainEnv(screen)
running = True
while running:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    screen.fill("white")
    main_env.draw([node_current, node_true, node_wrong, node_true, node_wrong])
    pg.display.flip()
    clock.tick(60)

pg.quit()
