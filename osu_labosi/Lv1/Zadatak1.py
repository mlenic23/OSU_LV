#Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je plaćen po radnom satu. 
#Koristite ugrađenu Python metodu input(). Nakon toga izračunajte koliko je korisnik zaradio i ispišite na ekran. 
#Na kraju prepravite rješenje na način da ukupni iznos izračunavate u zasebnoj funkciji naziva total_euro.
#Primjer:
    #Radni sati: 35 h
    #eura/h: 8.5
    #Ukupno: 297.5 eura

def total_euro(hours, euro_per_hour):

    return hours * euro_per_hour

hours = int(input("Unesite radne sate: "))
euro_per_hour = float(input("Unesite koliko ste plaćeni po satu: "))

print("Radni sati: ", hours)
print("eura/h: ", euro_per_hour)
print("Ukupno: ",total_euro(hours, euro_per_hour))

