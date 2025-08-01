import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import joblib

# Charger les données
df = pd.read_csv("DataSet_Emails.csv")
df["text"] = df["text"].fillna("")

# Charger le vectorizer + modèle
vectorizer, model = joblib.load("SVM_(Linear)_model.pkl")

# Titre
st.title("📧 Détection de Spam - Application Interactive")
st.markdown("Projet de Machine Learning : Test d'un email et visualisation des données.")

# =============== 1. Visualisation EDA =================
st.header("🔍 Visualisation des Données (EDA)")

# 📊 Répartition des classes
st.subheader("📊 Répartition des classes")
fig, ax = plt.subplots()
sns.countplot(data=df, x='label', palette='Set2')
plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])
plt.xlabel("Classe")
plt.ylabel("Nombre d'emails")
st.pyplot(fig)

# ☁️ WordCloud Ham
st.subheader("☁️ WordCloud des emails Ham")
ham_text = " ".join(df[df['label'] == 0]['text'])
ham_wc = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
fig2, ax2 = plt.subplots()
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig2)

# ☁️ WordCloud Spam (nouveau)
st.subheader("☁️ WordCloud des emails Spam")
spam_text = " ".join(df[df['label'] == 1]['text'])
spam_wc = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(spam_text)
fig3, ax3 = plt.subplots()
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig3)

# =============== 2. Tester un email ====================
st.header("✉️ Tester un Email")

email_input = st.text_area("Entrez le contenu de l'email ci-dessous :")

if st.button("🔍 Analyser"):
    if email_input.strip() == "":
        st.warning("⚠️ Veuillez saisir un email.")
    else:
        # Vectoriser et prédire
        X_input = vectorizer.transform([email_input])
        prediction = model.predict(X_input)[0]

        # Afficher résultat
        st.subheader("📌 Résultat de la prédiction :")
        if prediction == 1:
            st.error("🚫 Cet email est un *SPAM*.")
        else:
            st.success("✅ Cet email est un *HAM* (non-spam).")
