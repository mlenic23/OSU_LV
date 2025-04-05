import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay, accuracy_score, classification_report


labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("Lv5/penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy().ravel()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


#a)Pomoću stupčastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu pingvina) u skupu podataka za učenje i skupu podataka za testiranje. 
#Koristite numpy funkciju unique.
vals, train_count = np.unique(y_train, return_counts=True)
vals, test_count = np.unique(y_test, return_counts=True)

print(train_count)
print(test_count)

plt.figure()
plt.bar([0,1,2], [train_count[0], train_count[1], train_count[2]], width=0.4)
plt.bar([0+0.4,1+0.4,2+0.4], [test_count[0], test_count[1], test_count[2]], width=0.4)
plt.xticks([0,1,2],["Adelie", "Chinstrap", "Gentoo"])
plt.show()

#b)Izgradite model logističke regresije pomoću scikit-learn biblioteke na temelju skupa podatakaza učenje.
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

#c)Pronađite u atributima izgrađenog modela parametre modela. 
#Koja je razlika u odnosu na binarni klasifikacijski problem iz prvog zadatka?
print("Coefficientsi: ", log_reg_model.coef_)
print("Intercept: ", log_reg_model.intercept_)

#d)Pozovite funkciju plot_decision_region pri čemu joj predajte podatke za učenje i izgrađeni model logističke regresije. 
#Kako komentirate dobivene rezultate?
plot_decision_regions(X_train, y_train, log_reg_model)
plt.show()

#e)Provedite klasifikaciju skupa podataka za testiranje pomoću izgrađenog modela logističke regresije. 
#Izračunajte i prikažite matricu zabune na testnim podacima. Izračunajte točnost.
#Pomoću classification_report funkcije izračunajte vrijednost četiri glavne metrike na skupu podataka za testiranje.
y_test_p = log_reg_model.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()

print("Accuracy: ", accuracy_score(y_test, y_test_p))
print(classification_report(y_test, y_test_p))


#f)Dodajte u model još ulaznih veličina. 
#Što se događa s rezultatima klasifikacije na skupu podataka za testiranje?

input_variables = ['bill_length_mm',
                    'flipper_length_mm',
                    'bill_depth_mm',
                    'body_mass_g',]


X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy().ravel()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

y_test_p = log_reg_model.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()

print("Accuracy: ", accuracy_score(y_test, y_test_p))
print(classification_report(y_test, y_test_p))