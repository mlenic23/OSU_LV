#Napišite program koji ce kreirati sliku koja sadrži četiri kvadrata crne odnosno bijele boje (vidi primjer slike 2.4 ispod). 
#Za kreiranje ove funkcije koristite numpy funkcije zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. 
#Kako biste ih složili u odgovarajuci oblik koristite numpy funkcije hstack i vstack.

import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50,50))
white = np.ones((50,50))

row1 = np.hstack((black, white))
row2 = np.hstack((white, black))

board = np.vstack((row1, row2))

plt.imshow(board, cmap="gray")
plt.show()
