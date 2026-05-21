import numpy as np

class GridMap:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8) # 0 = free, 1 = occupied
        
        #Create some walls / obstacles 
        #Outer Walls 
        
        self.grid[0:5, :] = 1 
        self.grid[-5:, :] = 1 
        self.grid[:, 0:5] = 1
        self.grid[:, -5:] = 1
        
        # Internal walls
        self.grid[20:25, 20:70] = 1
        self.grid[40:80, 40:45] = 1
        self.grid[60:65, 60:90] = 1
        self.grid[30:70, 80:85] = 1
        
        # A few small obstacles
        self.grid[15:20, 80:85] = 1
        self.grid[75:80, 20:30] = 1
        
    def is_occupied(self, x, y):
        """Check if a point is occupied (wall)"""
        x, y = int(x), int(y)
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 1
        return True  # outside map = wall
    
    def get_map(self):
        return self.grid.copy()
        