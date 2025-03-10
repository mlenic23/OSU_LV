#Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i ženama. 
#Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja data pri cemu je u prvom stupcu polja oznaka spola (1 muško, 0 žensko), 
#drugi stupac polja je visina u cm, a treći stupac polja je masa u kg.

#a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja? 
#b) Prikažite odnos visine i mase osobe pomocu naredbe matplotlib.pyplot.scatter.
#c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
#d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom podatkovnom skupu.
#e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka. ind = (data[:,0] == 1)

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Lv2/data.csv", delimiter=",", skiprows=1)

num_people = data.shape[0]
print("Broj osoba: ", num_people)

plt.scatter(data[:,1], data[:,2], alpha=0.5, color="pink")
plt.xlabel("Visina")
plt.ylabel("Masa")
plt.title("Dijagram ovisnosti visine i mase")
plt.show()

plt.scatter(data[::50,1], data[::50,2], alpha=0.5, color="lightblue")
plt.xlabel("Visina")
plt.ylabel("Masa")
plt.title("Dijagram ovisnosti visine i mase za svaku 50. osobu")
plt.show()

min_height = np.min(data[:,1])
max_height = np.max(data[:,1])
avg_height = np.mean(data[:,1])

print("Min visina: ", min_height)
print("Max visina: ", max_height)
print("Avg visina: ", avg_height)

ind_m = (data[:,0] == 1)
min_height_m = np.min(data[ind_m,1])
max_height_m = np.max(data[ind_m,1])
avg_height_m = np.mean(data[ind_m,1])

print("Min visina (m): ", min_height_m)
print("Max visina (m): ", max_height_m)
print("Avg visina (m): ", avg_height_m)

ind_z = (data[:,0] == 0)
min_height_z = np.min(data[ind_z,1])
max_height_z = np.max(data[ind_z,1])
avg_height_z = np.mean(data[ind_z,1])

print("Min visina (ž): ", min_height_z)
print("Max visina (ž): ", max_height_z)
print("Avg visina (ž): ", avg_height_z)

