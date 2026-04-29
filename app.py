import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="MarketPulse E-Commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# FONCTIONS UTILITAIRES ET DONNÉES
# ==========================================
def generer_donnees_demo(n=200):
    """Génère un jeu de données simulé pour le mode démo."""
    np.random.seed(42)
    age = np.random.normal(35, 10, n).astype(int)
    heures_streaming = np.random.uniform(1, 40, n)
    score_engagement = (heures_streaming * 0.5) + (np.random.normal(0, 2, n))
    
    # Relation linéaire avec du bruit pour les achats
    achats_mensuels = 10 + (heures_streaming * 2.5) + (age * 0.5) + np.random.normal(0, 15, n)
    achats_mensuels = np.clip(achats_mensuels, 0, None) # Pas de valeurs négatives
    
    categories = ['Électronique', 'Mode', 'Alimentation', 'Divertissement']
    categorie_preferee = np.random.choice(categories, n)
    
    # Classification: Abonnement premium basé sur l'engagement
    prob_premium = 1 / (1 + np.exp(-(score_engagement - 10) / 3))
    abonnement_premium = np.random.binomial(1, prob_premium).astype(bool)
    
    df = pd.DataFrame({
        'Age': age,
        'Heures_Streaming_Hebdo': heures_streaming,
        'Score_Engagement': score_engagement,
        'Achats_Mensuels_Euros': achats_mensuels,
        'Categorie_Preferee': categorie_preferee,
        'Abonnement_Premium': abonnement_premium
    })
    
    # Introduction volontaire de quelques valeurs manquantes pour démontrer la robustesse
    df.loc[np.random.choice(df.index, 5), 'Age'] = np.nan
    return df

def preparer_donnees(df):
    """Gère les valeurs manquantes et encode les variables catégorielles."""
    df_clean = df.copy()
    
    # Remplacer les valeurs manquantes numériques par la médiane
    cols_num = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[cols_num] = df_clean[cols_num].fillna(df_clean[cols_num].median())
    
    # Encodage des catégories pour les modèles
    le = LabelEncoder()
    df_clean['Categorie_Code'] = le.fit_transform(df_clean['Categorie_Preferee'].astype(str))
    df_clean['Abonnement_Premium'] = df_clean['Abonnement_Premium'].astype(int)
    
    return df_clean

# Initialisation des données dans la session Streamlit
if 'data' not in st.session_state:
    st.session_state['data'] = generer_donnees_demo()

df_current = st.session_state['data']

# ==========================================
# BARRE DE NAVIGATION
# ==========================================
st.sidebar.title("MarketPulse 📡")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Choisissez une page :",
    ["Accueil", "Collecte de Données", "Dashboard Descriptif", 
     "Régression Simple", "Régression Multiple", 
     "Réduction de Dimension (ACP)", "Classification Supervisée", 
     "Clustering (Non Supervisé)"]
)

# Fonction d'export CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📥 Exporter les données (CSV)",
    data=convert_df(df_current),
    file_name='marketpulse_data.csv',
    mime='text/csv',
)

# ==========================================
# CONTENU DES PAGES
# ==========================================

if page == "Accueil":
    st.title("Bienvenue sur MarketPulse E-Commerce 🛒")
    st.markdown("""
    Cette application web analytique permet de collecter, stocker et analyser les habitudes 
    de consommation de contenu numérique et de commerce en ligne.
    
    **Fonctionnalités principales :**
    * 📝 **Collecte :** Formulaire interactif pour ajouter de nouvelles observations.
    * 📊 **Description :** Dashboard interactif avec indicateurs clés (KPIs).
    * 📈 **Prédiction & Modélisation :** Régressions, Classification, et Clustering.
    
    *Projet Universitaire - Master Informatique (INF 232 EC2)*
    """)
    st.info(f"Le jeu de données actuel contient **{len(df_current)}** observations.")

elif page == "Collecte de Données":
    st.title("📝 Collecte de Données Utilisateur")
    st.markdown("Remplissez ce formulaire pour ajouter une nouvelle observation au jeu de données.")
    
    with st.form("data_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", min_value=12, max_value=100, value=25)
            heures = st.slider("Heures de streaming hebdomadaire", 0.0, 100.0, 10.0)
            score = st.number_input("Score d'engagement (0-20)", min_value=0.0, max_value=20.0, value=10.0)
        with col2:
            achats = st.number_input("Achats mensuels estimés (€)", min_value=0.0, value=50.0)
            cat = st.selectbox("Catégorie préférée", ['Électronique', 'Mode', 'Alimentation', 'Divertissement'])
            premium = st.checkbox("Abonné Premium ?")
            
        submitted = st.form_submit_button("Enregistrer les données")
        
        if submitted:
            new_data = pd.DataFrame({
                'Age': [age], 'Heures_Streaming_Hebdo': [heures], 
                'Score_Engagement': [score], 'Achats_Mensuels_Euros': [achats],
                'Categorie_Preferee': [cat], 'Abonnement_Premium': [premium]
            })
            st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)
            st.success("Données ajoutées avec succès !")

elif page == "Dashboard Descriptif":
    st.title("📊 Dashboard d'Analyse Descriptive")
    df_clean = preparer_donnees(df_current)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moyenne d'Âge", f"{df_clean['Age'].mean():.1f} ans")
    col2.metric("Achats Moyens", f"{df_clean['Achats_Mensuels_Euros'].mean():.2f} €")
    col3.metric("Heures Streaming (Moy)", f"{df_clean['Heures_Streaming_Hebdo'].mean():.1f} h")
    col4.metric("Taux Premium", f"{(df_clean['Abonnement_Premium'].mean() * 100):.1f} %")
    
    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Répartition par Catégorie")
        fig1 = px.pie(df_clean, names='Categorie_Preferee', hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        st.subheader("Distribution des Achats")
        fig2 = px.histogram(df_clean, x='Achats_Mensuels_Euros', nbins=20, marginal="box")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Régression Simple":
    st.title("📈 Régression Linéaire Simple")
    st.markdown("Objectif : Prédire les achats mensuels en fonction des heures de streaming.")
    
    df_clean = preparer_donnees(df_current)
    
    try:
        X = df_clean[['Heures_Streaming_Hebdo']]
        y = df_clean['Achats_Mensuels_Euros']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Coefficient (Pente) :** {model.coef_[0]:.2f}")
            st.write(f"**Ordonnée à l'origine :** {model.intercept_:.2f}")
        with col2:
            st.write(f"**Erreur Quadratique Moyenne (MSE) :** {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"**Score R² :** {r2_score(y_test, y_pred):.2f}")
            
        fig = px.scatter(df_clean, x='Heures_Streaming_Hebdo', y='Achats_Mensuels_Euros', opacity=0.6)
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_range = model.predict(x_range)
        fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_range, mode='lines', name='Ligne de régression', line=dict(color='red')))
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")

elif page == "Régression Multiple":
    st.title("📉 Régression Linéaire Multiple")
    st.markdown("Objectif : Prédire les achats en combinant plusieurs facteurs (Âge, Streaming, Engagement).")
    
    df_clean = preparer_donnees(df_current)
    features = st.multiselect("Sélectionnez les variables explicatives :", 
                              ['Age', 'Heures_Streaming_Hebdo', 'Score_Engagement'], 
                              default=['Age', 'Heures_Streaming_Hebdo'])
    
    if len(features) >= 2:
        try:
            X = df_clean[features]
            y = df_clean['Achats_Mensuels_Euros']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.success(f"Modèle entraîné avec succès ! Score R² : {r2_score(y_test, y_pred):.2f}")
            
            # Visualisation Valeurs Réelles vs Prédites
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Valeurs Réelles (€)', 'y': 'Valeurs Prédites (€)'})
            fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur lors de la modélisation : {e}")
    else:
        st.warning("Veuillez sélectionner au moins 2 variables.")

elif page == "Réduction de Dimension (ACP)":
    st.title("🎯 Analyse en Composantes Principales (ACP)")
    st.markdown("Objectif : Réduire la dimensionnalité pour visualiser les profils utilisateurs en 2D.")
    
    df_clean = preparer_donnees(df_current)
    cols_num = ['Age', 'Heures_Streaming_Hebdo', 'Score_Engagement', 'Achats_Mensuels_Euros']
    
    try:
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[cols_num])
        
        # ACP
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        
        df_pca = pd.DataFrame(data=components, columns=['CP1', 'CP2'])
        df_pca['Abonnement_Premium'] = df_clean['Abonnement_Premium'].astype(str)
        
        st.write(f"**Variance expliquée par CP1 :** {pca.explained_variance_ratio_[0]*100:.2f}%")
        st.write(f"**Variance expliquée par CP2 :** {pca.explained_variance_ratio_[1]*100:.2f}%")
        
        fig = px.scatter(df_pca, x='CP1', y='CP2', color='Abonnement_Premium', 
                         title="Projection des utilisateurs sur les 2 premières composantes")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur ACP : {e}")

elif page == "Classification Supervisée":
    st.title("🤖 Classification Supervisée")
    st.markdown("Objectif : Prédire si un utilisateur va souscrire à un Abonnement Premium.")
    
    df_clean = preparer_donnees(df_current)
    
    try:
        X = df_clean[['Age', 'Heures_Streaming_Hebdo', 'Score_Engagement', 'Achats_Mensuels_Euros']]
        y = df_clean['Abonnement_Premium']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardisation requise pour la régression logistique
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        st.metric("Précision (Accuracy) du Modèle", f"{acc*100:.2f}%")
        
        # Matrice de confusion simplifiée
        st.text("Rapport de Classification :")
        st.code(classification_report(y_test, y_pred))
        
    except Exception as e:
        st.error(f"Erreur de classification : {e}")

elif page == "Clustering (Non Supervisé)":
    st.title("🧩 Classification Non Supervisée (Clustering)")
    st.markdown("Objectif : Découvrir des segments naturels d'utilisateurs avec l'algorithme K-Means.")
    
    df_clean = preparer_donnees(df_current)
    k = st.slider("Choisissez le nombre de clusters (K) :", 2, 6, 3)
    
    try:
        X = df_clean[['Heures_Streaming_Hebdo', 'Achats_Mensuels_Euros']]
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_clean['Cluster'] = kmeans.fit_predict(X_scaled)
        
        fig = px.scatter(df_clean, x='Heures_Streaming_Hebdo', y='Achats_Mensuels_Euros', 
                         color=df_clean['Cluster'].astype(str),
                         title=f"Segmentation K-Means (K={k})")
        st.plotly_chart(fig, use_container_width=True)
        
        # Profil moyen des clusters
        st.subheader("Profil moyen par Cluster")
        cluster_profiles = df_clean.groupby('Cluster')[['Heures_Streaming_Hebdo', 'Achats_Mensuels_Euros', 'Age']].mean()
        st.dataframe(cluster_profiles)
        
    except Exception as e:
        st.error(f"Erreur de clustering : {e}")