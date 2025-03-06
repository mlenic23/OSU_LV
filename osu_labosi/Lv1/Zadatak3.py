#Napišite program koji od korisnika zahtijeva unos brojeva u beskonačnoj petlji sve dok korisnik ne upiše „Done“ (bez navodnika). 
#Pri tome brojeve spremajte u listu. Nakon toga potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu vrijednost. 
#Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
#(npr. slovo umjesto brojke) na način da program zanemari taj unos i ispiše odgovarajuću poruku.

lista = []

while True:
    unos = input("")

    if unos == 'Done':
        break

    try:
        broj = float(unos)
        lista.append(broj)
        
    except ValueError:
        print("Molim vas unesite broj!")

zbroj = sum(lista)
elementi = len(lista)

srednja_vrijednost = zbroj / elementi
minimalna = min(lista)
maximalna = max(lista)

print("Ukupan broj unosa: ",elementi)
print("Srednja vrijednost: ",srednja_vrijednost)
print("Minimalna vrijednost: ",minimalna)
print("Maximalna vrijednost: ",maximalna)

lista.sort()
print("Sortirana lista: ",lista)