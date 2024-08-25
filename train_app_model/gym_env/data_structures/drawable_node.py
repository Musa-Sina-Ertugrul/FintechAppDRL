import pygame as pg
from .node import Node

class DrawableNode(Node):
    
    font : pg.font.Font = pg.font.SysFont()
    def __init__(self, name: str, number: int, is_wrong: bool, id: int,surface:pg.Surface) -> None:
        super().__init__(name, number, is_wrong, id)
        self.__surface = surface
        self.__pygame_text = self.font.render(self.name,antialias=0,color='black',background=(0,0,0,0))
        
    
    def draw(self,x:int,y:int) -> None:
        center_x , center_y = x-20,y-10
        rect = pg.Rect(center_x,center_y,40,20)
        if self:
            ellipse : pg.Rect = pg.draw.ellipse(self.__surface,'green',rect)
        else: 
            ellipse : pg.Rect = pg.draw.ellipse(self.__surface,'red',rect)
        self.__surface.blit(self.__pygame_text,dest=(center_x,center_y))
        self.__surface.blit(ellipse,dest=(center_x,center_y))
        
