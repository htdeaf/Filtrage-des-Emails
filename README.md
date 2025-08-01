# 📧 Détection de Spam - Application Interactive

## Description

Ce projet propose une application web interactive développée avec **Streamlit** permettant de détecter si un email est un spam ou un message légitime (ham).
L’application inclut une visualisation exploratoire des données (EDA) ainsi qu’un module de test pour analyser un email en temps réel à l’aide d’un modèle de machine learning (Linear SVM).

---

## Fonctionnalités

### 1. Visualisation des données (EDA)

* Affichage de la répartition des emails spam et ham sous forme de graphique à barres.
* Nuages de mots (WordCloud) pour visualiser les mots les plus fréquents dans les emails spam et ham.

### 2. Test d’un email

* Interface simple pour saisir un email à analyser.
* Prédiction instantanée du type (spam ou ham) via un modèle Linear SVM pré-entraîné.
* Affichage clair du résultat avec une notification colorée.

---

## Installation

1. Cloner ce dépôt :

```bash
git clone <url_du_depot>
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Utilisation

1. Lancer l’application Streamlit :

```bash
streamlit run app.py
```

2. Explorer les visualisations et tester des emails via l’interface web.

---

## Fichiers importants

* `app.py` : script principal de l’application Streamlit.
* `DataSet_Emails.csv` : fichier de données contenant les emails annotés (spam/ham).
* `SVM_(Linear)_model.pkl` : modèle Linear SVM pré-entraîné et le vectorizer sauvegardés avec joblib.

---

## Modèle Machine Learning

* **Algorithme** : Linear Support Vector Machine (SVM)
* **Prétraitement** : Vectorisation TF-IDF des emails
* **Performance** : F1-score pondéré ≈ 0.99
* **Optimisation** : hyperparamètres réglés via GridSearchCV pour maximiser la précision.

---

## Auteurs

* Salma Rehmi — Étudiante Chercheuse en Machine Learning

---

## Remarques

* L’application ne fonctionne qu’avec des emails en anglais (données d’entraînement).
* Pour de meilleurs résultats, veillez à saisir un texte clair et compréhensible.

---
