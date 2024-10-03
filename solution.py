

import numpy as np
import matplotlib.pyplot as plt


# Chargement de l'ensemble de données
iris = np.genfromtxt('iris.txt')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

class Q1:

    def feature_means(self, iris):
        return np.mean(iris[:,:-1],axis=0)

    def empirical_covariance(self, iris):
        return np.cov(iris[:,:-1], rowvar=False)

    def feature_means_class_1(self, iris):
        iris_class_1 = iris[iris[:,-1]==1]
        return np.mean(iris_class_1[:,:-1],axis=0)

    def empirical_covariance_class_1(self, iris):
        iris_class_1 = iris[iris[:,-1]==1]
        return np.cov(iris_class_1[:,:-1],rowvar=False)

# Création d'une instance de la classe Q1
q1_instance = Q1()

# Appel de la méthode feature_means avec les données iris
means = q1_instance.feature_means(iris)

# Afficher le résultat
print("Les moyennes des attributs sont :", means)

# Appel de la méthode empirical_covariance avec les données iris
cov = q1_instance.empirical_covariance(iris)

# Afficher le résultat
print("La matrice de covariance des attributs est :", cov)

# Appel de la méthode feature_means_class_1 avec les données iris
means_class_1 = q1_instance.feature_means_class_1(iris)

# Afficher le résultat
print("Les moyennes des attributs pour les points dont le label est 1 :", means_class_1)

# Appel de la méthode empirical_covariance_class_1 avec les données iris
cov_class_1 = q1_instance.empirical_covariance_class_1(iris)

# Afficher le résultat
print("La matrice de covariance des attributs pour les points dont le label est 1 :", cov_class_1)

def dist_L1(x, Y):
    "La distance L1 (Manhattan)"
    return np.sum((np.abs(x - Y)), axis=1)

class HardParzen:
    def __init__(self, h):
        self.h = h

    def fit(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)

    def predict(self, test_data):
        num_test = test_data.shape[0]
        class_pred = [] # initialisation de la liste des prédictions

        # Pour chaque point de données de test
        for i in range(num_test):
            # Déterminer les distances par rapport à chaque point de l'ensemble d'apprentissage
            distances = dist_L1(test_data[i,:], self.train_inputs)
            # Trouver les indices des données x d'entraîment qui vérifient la condition d(x, test_data) < h
            ind_neighbors = [j for j in range(len(distances)) if distances[j] < self.h]
            class_neigbors = self.train_labels[ind_neighbors] # Les labels des voisins

            # Cas où x n'a pas de voisin
            if list(class_neigbors) ==[]:
                class_pred.append(draw_rand_label(test_data[i,:], self.label_list))

            else:
                # On recupère le plus proche voisin
                class_pred.append(max(list(class_neigbors), key=list(class_neigbors).count))

        return np.array(class_pred)

# Tester le modèle de Hard Parzen
h = 0.5
f = HardParzen(h)
X_train = iris[:,:-1]
y_train = iris[:,-1]
f.fit(X_train,y_train)
y_pred = f.predict(X_train) # Prédictions sur l'ensemble d'entraînement
erreur_moy = np.mean(y_train != y_pred) # Evaluation du modèle sur l'ensemble d'entraînement
print(erreur_moy)

class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)

    def rbf_kernel(self, x, y):
        """
        Calcule le noyau RBF entre deux vecteurs en utilisant la distance L1.
        x: np.array de taille (n_features,)
        y: np.array de taille (n_features,)
        """
        # Distance L1 (Manhattan)
        dist = np.sum(np.abs(x - y))
        d = x.shape[0]  # Dimension des points x et y

        # Calcul de la densité
        coefficient = 1 / ((2 * np.pi) ** (d / 2) * self.sigma ** d)
        exponent = np.exp(-0.5 * dist ** 2 / self.sigma ** 2)

        return coefficient * exponent

    def one_hot(self, y_i, num_classes):
        """"
        la fonction one-hot encoding
        y_i : un label (1, 2,... ou m)
        num_classes : le nombre de classes
        """
        one_hot_vector = np.zeros(num_classes)
        one_hot_vector[int(y_i) - 1] = 1

        return one_hot_vector

    def predict(self, test_data):
        num_test = test_data.shape[0]
        num_train = self.train_inputs.shape[0]
        y_pred = np.zeros(num_test)
        eps = 1e-10  # Petite constante pour éviter la division par 0
        weights = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                # Calculer les poids RBF
                weights[i, j] = self.rbf_kernel(test_data[i, :], self.train_inputs[j, :])

            sum_weights = np.sum(weights[i, :])  # Somme des poids pour le point test i

            if sum_weights < eps:
                # Si la somme des poids est proche de 0, éviter la division
                y_pred[i] = draw_rand_label(test_data[i,:], self.label_list)# une prédiction alèatoire
            else:
                # One-hot encode pour le point d'entraînement correspondant
                num_classes = len(set(self.train_labels))
                one_hot = np.zeros((num_train, num_classes))

                for j in range(num_train):
                    one_hot[j, :] = self.one_hot(self.train_labels[j], num_classes)

                # Calculer la prédiction avec la somme pondérée des vecteurs one-hot
                weighted_sum = np.sum(weights[i, :].reshape(-1, 1) * one_hot, axis=0)

                # Normalisation avec sum_weights et éviter la division par 0
                y_pred[i] = np.argmax(weighted_sum / sum_weights) + 1  # +1 car les labels commencent à 1

        return y_pred

# Tester le modèle Soft Parzen
# Création du modèle
model = SoftRBFParzen(sigma = 1)

# Entraînement du modèle
X_train = iris[:,:-1]
y_train = iris[:,-1]
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_train)

print(np.mean(y_train != y_pred)) # Evaluation du modèle sur l'ensemble d'entraînement

def split_dataset(iris):
    """ Cette fonction permet de séparer le dataset en trois ensembles
    train, validion et test de la façon suivante décrite à la question 4 """
    num_iris = iris.shape[0]

    # Calcule les index des ensembles
    index_train = [j for j in range(num_iris) if (j%5 == 0 or j%5 == 1 or j%5 == 2)]
    index_val = [j for j in range(num_iris) if j%5 == 3]
    index_test = [j for j in range(num_iris) if j%5 == 4]

    # Train, validation et test
    train = iris[index_train]
    validation = iris[index_val]
    test = iris[index_test]

    return (train, validation, test)

class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        parzen_model = HardParzen(h)
        parzen_model.fit(self.x_train, self.y_train)
        y_pred = parzen_model.predict(self.x_val)
        erreur = np.mean(self.y_val != y_pred)

        return erreur

    def soft_parzen(self, sigma):
        SoftRBFParzen_model = SoftRBFParzen(sigma)
        SoftRBFParzen_model.fit(self.x_train, self.y_train)
        y_pred = SoftRBFParzen_model.predict(self.x_val)
        erreur = np.mean(self.y_val != y_pred)

        return erreur

# Faire appel les fonctions ErrorRate.hard_parzen et ErrorRate.soft_parzen
a = split_dataset(iris) # Séparer les données en (train, val, test)
x_train = a[0][:,:-1]
y_train = a[0][:,-1]
x_val = a[1][:,:-1]
y_val = a[1][:,-1]
x_test = a[2][:,:-1]
y_test = a[2][:,-1]
f = ErrorRate(x_train, y_train, x_val, y_val)
h = 1
sigma = 1
print(f"Taux d'erreur HardParzen: {f.hard_parzen(h)}")
print(f"Taux d'erreur SoftParzen: {f.soft_parzen(sigma)}")

import matplotlib.pyplot as plt

abcisses = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
errors_hardparzen = [f.hard_parzen(h) for h in abcisses]
errors_softparzen = [f.soft_parzen(sigma) for sigma in abcisses]

# Plotting the errors
plt.plot(abcisses, errors_hardparzen, label='Hard Parzen', marker='o')
plt.plot(abcisses, errors_softparzen, label='Soft Parzen', marker='o')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Errors')
plt.title('Error Comparaison: Hard Parzen vs Soft Parzen')

# Display the legend and the plot
plt.legend()
plt.xscale('log')  # Utilisation d'une échelle logarithmique pour une meilleure vision
plt.show()



# Find h* and sigma*
h_opt = abcisses[np.argmin(errors_hardparzen)]
sigma_opt = abcisses[np.argmin(errors_softparzen)]
print("h* =",h_opt,"et", "sigma* =", sigma_opt)

def get_test_errors(iris):
    train, validation, test = split_dataset(iris)
    x_train = train[:,:-1]
    y_train = train[:,-1]
    x_test = test[:,:-1]
    y_test = test[:,-1]

    error_instance =  ErrorRate(x_train, y_train, x_test, y_test)
    error_hardparzen = error_instance.hard_parzen(h_opt)
    error_softparzen = error_instance.soft_parzen(sigma_opt)

    return np.array([error_hardparzen, error_softparzen])

# Complexité temporelle des deux méthodes
import time
class measure_time():

    def hard_parzen(self, h):
        parzen_model = HardParzen(h)

        start_time = time.time()
        parzen_model.fit(x_train, y_train)
        y_pred = parzen_model.predict(x_test)
        end_time = time.time()

        return end_time - start_time

    def soft_parzen(self, sigma):
        SoftRBFParzen_model = SoftRBFParzen(sigma)

        start_time = time.time()
        SoftRBFParzen_model.fit(x_train, y_train)
        y_pred = SoftRBFParzen_model.predict(x_val)
        end_time = time.time()

        return end_time - start_time

# Comparer les temps de calcul pour différentes valeurs de h et sigma
time_instance = measure_time()
print("Hard Parzen Time Complexity:")
for h in abcisses:
    hard_parzen_time = time_instance.hard_parzen(h)
    print(f"h = {h}, Time: {hard_parzen_time:.4f} seconds")

print("\nSoft Parzen Time Complexity:")
for sigma in abcisses:
    soft_parzen_time = time_instance.soft_parzen(sigma)
    print(f"sigma = {sigma}, Time: {soft_parzen_time:.4f} seconds")

# Visualisation de la courbe des temps d'exécutions en fonction des valeurs de h ou sigma
hardparzen_times = [time_instance.hard_parzen(h) for h in abcisses]
softparzen_times = [time_instance.soft_parzen(sigma) for sigma in abcisses]

# Plotting the errors
plt.plot(abcisses, hardparzen_times, label='Hard Parzen', marker='o')
plt.plot(abcisses, softparzen_times, label='Soft Parzen', marker='o')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Times of execution')
plt.title('Times Comparaison: Hard Parzen vs Soft Parzen')

# Display the legend and the plot
plt.legend()
plt.show()



def random_projections(X, A):
    return (1 / np.sqrt(2)) * np.dot(X, A)

np.random.seed(42) # Pour travailler avec une seule matrice A fixe générée alèatoirement
A = np.random.rand(4, 2)  # Génère une matrice de projection A de dimension (4, 2)
X_proj = random_projections(iris[:,:-1], A)

def validation_errors(iris, num_projections, values, model):
    # values est une liste des valeurs de h ou sigma
    train, validation, test = split_dataset(iris)
    x_train = train[:,:-1]
    y_train = train[:,-1]
    x_val = validation[:,:-1]
    y_val = validation[:,-1]
    errors = np.zeros((500,10))

    for i in range(num_projections):
        for j in range(len(values)):
            # Générer une matrice A avec des valeurs tirées d'une gaussienne N(0, 1)
            A = np.random.randn(4, 2)
            # Projections
            x_train_proj = random_projections(x_train, A)
            x_val_proj = random_projections(x_val, A)

            errors_instance = ErrorRate(x_train_proj, y_train, x_val_proj, y_val)
            if model ==  "HardParzen" :
                errors[i,j] = errors_instance.hard_parzen(values[j])
            elif model == "SoftRBFParze" :
                errors[i,j] = errors_instance.soft_parzen(values[j])
    return errors

# Erreurs des deux méthodes sur l'ensemble de validation
errors_hardparzen = validation_errors(iris, num_projections=500, values=abcisses, model ="HardParzen")
errors_softparzen = validation_errors(iris, num_projections=500, values=abcisses, model = "SoftParzen")

# Calculer les moyennes des erreurs pour chaque valeur de h
mean_errors_hardparzen = np.mean(errors_hardparzen, axis=0)
mean_errors_softparzen = np.mean(errors_softparzen, axis=0)

# Calculer les écart-types des erreurs pour chaque valeur de h
std_errors_hardparzen = np.std(errors_hardparzen, axis=0)
std_errors_softparzen = np.std(errors_softparzen, axis=0)

# Calculer les intervalles d'erreur (0.2 * écart-type)
error_intervals_hardparzen = 0.2 * std_errors_hardparzen
error_intervals_softparzen = 0.2 * std_errors_softparzen

# Tracer les résultats
plt.errorbar(abcisses, mean_errors_hardparzen, yerr=error_intervals_hardparzen, fmt='-o', capsize=5, label='Erreur de validation moyenne de Hard Parzen')
plt.errorbar(abcisses, mean_errors_softparzen, yerr=error_intervals_softparzen, fmt='-o', capsize=5, label='Erreur de validation moyenne de Soft Parzen')

# Ajouter des légendes et des labels
plt.xlabel('Valeur de h et sigma')
plt.ylabel('Erreur de validation moyenne')
plt.title('Erreur de validation moyenne avec intervalles d\'erreur')
plt.grid(True)
plt.legend()
plt.xscale('log')  # Utilisation d'une échelle logarithmique pour une meilleure vision

# Afficher le graphique
plt.show()

