import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import pygame as pg
pg.init()
import sys
sys.path.append("/home/musasina/projects/FintechAppDRL/train_app_model")
from gym_env.data_structures import DrawableNode
from gym_env.data_structures.node_type import NodeType

screen = pg.display.set_mode((1280,720))
clock = pg.time.Clock()

node_true = DrawableNode("test1",-1,NodeType.TRUE,-1,screen)
node_wrong = DrawableNode("test2",-1,NodeType.WRONG,-1,screen)
running = True
while running:
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    screen.fill("white")
    node_true.draw(0,0)
    node_wrong.draw(100,100)
    pg.display.flip()
    clock.tick(60)
    
pg.quit()
        