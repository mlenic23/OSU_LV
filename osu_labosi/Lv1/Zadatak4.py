#Napišite Python skriptu koja će učitati tekstualnu datoteku naziva song.txt.
#Potrebno je napraviti rječnik koji kao ključeve koristi sve različite riječi koje se pojavljuju u datoteci, dok su vrijednosti jednake broju puta koliko se svaka riječ (ključ) pojavljuje u datoteci.
#Koliko je riječi koje se pojavljuju samo jednom u datoteci? Ispišite ih.


with open("Lv1/song.txt") as f:

    words = []

    lines = f.readlines()

    for line in lines:
        line = line.strip()  
        line_words = line.split(" ") 

        for word in line_words:
            cleaned_word = word.rstrip(",").lower()  
            words.append(cleaned_word) 

mapa = {}
for w in words:
    if w not in mapa:
        mapa[w] = 1
    else:
        mapa[w] += 1

single = set()
for k,v in mapa.items():
    if v == 1:
        single.add(k)

print("Riječi koje se pojavljuju samo jednom: \n", "\n ", single)
print("Ukupno riječi koje se pojavljuju samo jednom: ", len(single))