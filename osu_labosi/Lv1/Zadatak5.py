#Napišite Python skriptu koja će učitati tekstualnu datoteku naziva SMSSpamCollection.txt
#Ova datoteka sadrži 5574 SMS poruka pri čemu su neke označene kao spam, a neke kao ham.
#Primjer dijela datoteke:
#ham Yup next stop.
#ham Ok lar... Joking wif u oni...
#spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!

#a) Izračunajte koliki je prosječan broj riječi u SMS porukama koje su tipa ham, a koliko je prosječan broj riječi u porukama koje su tipa spam.
#b) Koliko SMS poruka koje su tipa spam završava uskličnikom ?

spam = []
ham = []

with open("Lv1/SMSSPamCollection.txt", encoding="utf8") as f:

    lines = [line.strip() for line in f.readlines()]  #strip svaku liniju koju pročita

    for line in lines:
        if line.startswith("spam"):
            spam.append(line[5:].strip())
        else:
            ham.append(line[4:].strip())

    spam_words = 0
    for s in spam:
        swords = s.split(" ")
        num_swords = len(swords)
        spam_words += num_swords
    avg_spam = spam_words / len(spam)
    print(avg_spam)

    ham_words = 0
    for h in ham:
        hwords = h.split(" ")
        num_hwords = len(hwords)
        ham_words += num_hwords
    avg_ham = ham_words / len(ham)
    print(avg_ham)

    spam_uskl = 0
    for s in spam:
        if s[-1] == "!":
            spam_uskl += 1
    print(spam_uskl)