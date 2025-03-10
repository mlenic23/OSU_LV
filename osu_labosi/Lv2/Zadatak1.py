#Pomoću funkcija numpy.array i matplotlib.pyplot pokušajte nacrtati sliku 2.3 u okviru skripte zadatak_1.py. 
# Igrajte se sa slikom, promijenite boju linija, debljinu linije i sl.

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 3, 1])
y = np.array([1, 2, 2, 1, 1])

plt.plot(x,y, 'pink', linewidth=1, marker=".", markersize=5)
plt.axis([0,4,0,4])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')
plt.show()
