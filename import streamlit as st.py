import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats

st.set_page_config(page_title="AgriSight Analytics | INF 232", page_icon="\U0001F33E", layout="wide")

if 'donnees' not in st.session_state:
    st.session_state.donnees = pd.DataFrame(columns=[
        'ID_Parcelle', 'Date', 'Temperature_C', 'Humidite_Pct',
        'pH_Sol', 'Precipitations_mm', 'Engrais_kg_ha', 'Rendement_t_ha'
    ])

with st.sidebar:
    st.title("AgriSight")
    st.markdown("**INF 232 EC2 - TP1**")
    st.markdown("---")
    
    st.subheader("Collecter de nouvelles donnees")
    with st.form("formulaire_saisie"):
        parcelle = st.text_input("ID Parcelle", "P01")
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("Temperature (C)", -10.0, 55.0, 25.0, 0.1)
            humidite = st.slider("Humidite (%)", 0, 100, 60)
            ph = st.number_input("pH du Sol", 3.0, 10.0, 6.5, 0.1)
        with col2:
            date = st.date_input("Date")
            pluie = st.number_input("Precipitations (mm)", 0.0, 500.0, 10.0, 0.5)
            engrais = st.number_input("Engrais (kg/ha)", 0.0, 1000.0, 150.0, 5.0)
            rendement = st.number_input("Rendement (t/ha)", 0.0, 20.0, 5.5, 0.1)
        
        soumis = st.form_submit_button("Ajouter l'enregistrement")
        if soumis:
            nouvelle_ligne = pd.DataFrame([{
                'ID_Parcelle': parcelle, 'Date': str(date), 'Temperature_C': temp,
                'Humidite_Pct': humidite, 'pH_Sol': ph, 'Precipitations_mm': pluie,
                'Engrais_kg_ha': engrais, 'Rendement_t_ha': rendement
            }])
            st.session_state.donnees = pd.concat([st.session_state.donnees, nouvelle_ligne], ignore_index=True)
            st.success(f"Enregistrement {parcelle} ajoute. Total : {len(st.session_state.donnees)}")

    st.markdown("---")
    st.subheader("Operations groupees")
    fichier_importe = st.file_uploader("Importer un fichier CSV", type=['csv'])
    if fichier_importe is not None:
        try:
            donnees_importees = pd.read_csv(fichier_importe)
            colonnes_requises = ['Temperature_C', 'Humidite_Pct', 'pH_Sol', 'Precipitations_mm', 'Engrais_kg_ha', 'Rendement_t_ha']
            if all(col in donnees_importees.columns for col in colonnes_requises):
                st.session_state.donnees = pd.concat([st.session_state.donnees, donnees_importees], ignore_index=True)
                st.success(f"{len(donnees_importees)} enregistrements importes.")
            else:
                st.error("Le CSV doit contenir : " + ", ".join(colonnes_requises))
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")

    if not st.session_state.donnees.empty:
        csv = st.session_state.donnees.to_csv(index=False).encode('utf-8')
        st.download_button("Telecharger les donnees", csv, "agrisight_donnees.csv", "text/csv")

st.title("Plateforme d'Analyse AgriSight")
st.markdown("**INF 232 EC2** - Collecte et analyse descriptive des donnees agricoles")
st.markdown("---")

if st.session_state.donnees.empty:
    st.info("Ajoutez des donnees via le formulaire ou importez un fichier CSV.")
    
    if st.button("Charger les donnees de demonstration"):
        np.random.seed(42)
        n = 150
        donnees_demo = pd.DataFrame({
            'ID_Parcelle': [f'P{str(i).zfill(2)}' for i in np.random.randint(1, 15, n)],
            'Date': [str(d) for d in pd.date_range('2026-01-01', periods=n, freq='5D')],
            'Temperature_C': np.random.normal(28, 4, n),
            'Humidite_Pct': np.random.normal(65, 15, n),
            'pH_Sol': np.random.normal(6.5, 0.8, n),
            'Precipitations_mm': np.random.exponential(50, n),
            'Engrais_kg_ha': np.random.normal(200, 50, n),
            'Rendement_t_ha': np.random.normal(6, 1.5, n)
        })
        donnees_demo['Rendement_t_ha'] = (2.5 + 0.1 * donnees_demo['Temperature_C'] - 0.02 * donnees_demo['Humidite_Pct'] + 0.5 * donnees_demo['pH_Sol'] + 0.01 * donnees_demo['Precipitations_mm'] + 0.02 * donnees_demo['Engrais_kg_ha'] + np.random.normal(0, 0.5, n)).clip(1.0, 12.0)
        st.session_state.donnees = donnees_demo
        st.rerun()

else:
    df = st.session_state.donnees.copy()
    
    onglet1, onglet2, onglet3, onglet4, onglet5 = st.tabs([
        "Statistiques Descriptives",
        "Regression Simple",
        "Regression Multiple",
        "ACP",
        "Classification"
    ])
    
    with onglet1:
        st.subheader("Donnees brutes")
        st.dataframe(df.head(10), use_container_width=True)
        st.metric("Nombre d'echantillons", len(df))
        st.metric("Rendement moyen (t/ha)", f"{pd.to_numeric(df['Rendement_t_ha']).mean():.2f}")
        st.subheader("Resume statistique")
        st.dataframe(df.describe(), use_container_width=True)
        
        variable = st.selectbox("Variable pour l'histogramme",
            ['Temperature_C', 'Humidite_Pct', 'pH_Sol', 'Precipitations_mm', 'Engrais_kg_ha', 'Rendement_t_ha'])
        fig = px.histogram(df, x=variable, nbins=25, marginal="box", title=f"Distribution de {variable}")
        st.plotly_chart(fig, use_container_width=True)
        
        colonnes_num = ['Temperature_C', 'Humidite_Pct', 'pH_Sol', 'Precipitations_mm', 'Engrais_kg_ha', 'Rendement_t_ha']
        df_num = df[colonnes_num].apply(pd.to_numeric, errors='coerce')
        fig_corr = px.imshow(df_num.corr(), text_auto=".2f", title="Matrice de correlation", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

    with onglet2:
        st.subheader("Regression lineaire simple")
        var_independante = st.selectbox("Variable independante (X)",
            ['Temperature_C', 'Humidite_Pct', 'pH_Sol', 'Precipitations_mm', 'Engrais_kg_ha'])
        
        X = pd.to_numeric(df[var_independante], errors='coerce').values.reshape(-1, 1)
        y = pd.to_numeric(df['Rendement_t_ha'], errors='coerce').values
        masque = ~np.isnan(X.flatten()) & ~np.isnan(y)
        X, y = X[masque], y[masque]
        
        if len(X) > 1:
            modele = LinearRegression()
            modele.fit(X, y)
            y_pred = modele.predict(X)
            r2 = r2_score(y, y_pred)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Coefficient", f"{modele.coef_[0]:.4f}")
            col2.metric("Intercept", f"{modele.intercept_:.2f}")
            col3.metric("R2", f"{r2:.3f}")
            
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_range = modele.predict(x_range)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Donnees'))
            fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_range, mode='lines', name=f'Regression (R2={r2:.3f})'))
            fig.update_layout(title=f"Rendement vs {var_independante}", xaxis_title=var_independante, yaxis_title="Rendement (t/ha)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Ajoutez plus de donnees.")

    with onglet3:
        st.subheader("Regression lineaire multiple")
        colonnes_caract = ['Temperature_C', 'Humidite_Pct', 'pH_Sol', 'Precipitations_mm', 'Engrais_kg_ha']
        df_model = df[colonnes_caract + ['Rendement_t_ha']].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df_model) > len(colonnes_caract):
            X_multi = df_model[colonnes_caract].values
            y_multi = df_model['Rendement_t_ha'].values
            X_entr, X_test, y_entr, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
            modele_multi = LinearRegression()
            modele_multi.fit(X_entr, y_entr)
            y_pred_multi = modele_multi.predict(X_test)
            r2_multi = r2_score(y_test, y_pred_multi)
            st.success(f"R2 sur le test : {r2_multi:.3f}")
            
            df_coef = pd.DataFrame({'Variable': colonnes_caract, 'Coefficient': modele_multi.coef_})
            st.dataframe(df_coef, use_container_width=True)
            
            fig_reel = px.scatter(x=y_test, y=y_pred_multi, labels={'x': 'Reel', 'y': 'Predit'}, title="Reel vs Predit")
            fig_reel.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(dash='dash', color='red'))
            st.plotly_chart(fig_reel, use_container_width=True)
        else:
            st.warning(f"Il faut au moins {len(colonnes_caract)+1} echantillons. Actuellement : {len(df_model)}.")

    with onglet4:
        st.subheader("Analyse en Composantes Principales")
        colonnes_acp = ['Temperature_C', 'Humidite_Pct', 'pH_Sol', 'Precipitations_mm', 'Engrais_kg_ha']
        df_acp = df[colonnes_acp + ['Rendement_t_ha', 'ID_Parcelle']].copy()
        df_acp[colonnes_acp] = df_acp[colonnes_acp].apply(pd.to_numeric, errors='coerce')
        df_acp = df_acp.dropna()
        
        if len(df_acp) > 2:
            caracteristiques = df_acp[colonnes_acp].values
            caracteristiques_centrees = StandardScaler().fit_transform(caracteristiques)
            acp = PCA(n_components=2)
            composantes = acp.fit_transform(caracteristiques_centrees)
            df_acp['CP1'] = composantes[:, 0]
            df_acp['CP2'] = composantes[:, 1]
            
            st.metric("Variance expliquee CP1", f"{acp.explained_variance_ratio_[0]:.2%}")
            st.metric("Variance expliquee CP2", f"{acp.explained_variance_ratio_[1]:.2%}")
            
            fig_acp = px.scatter(df_acp, x='CP1', y='CP2', color='Rendement_t_ha', hover_data=['ID_Parcelle'], title="Visualisation ACP", color_continuous_scale='Viridis')
            st.plotly_chart(fig_acp, use_container_width=True)
        else:
            st.warning("Ajoutez plus de donnees pour l'ACP.")

    with onglet5:
        st.subheader("Classification K-Means")
        colonnes_cluster = ['Temperature_C', 'Humidite_Pct', 'pH_Sol', 'Precipitations_mm', 'Engrais_kg_ha']
        df_clust = df[colonnes_cluster + ['Rendement_t_ha', 'ID_Parcelle']].copy()
        df_clust[colonnes_cluster] = df_clust[colonnes_cluster].apply(pd.to_numeric, errors='coerce')
        df_clust = df_clust.dropna()
        
        if len(df_clust) > 3:
            n_clusters = st.slider("Nombre de clusters", 2, 5, 3)
            caracteristiques_centrees = StandardScaler().fit_transform(df_clust[colonnes_cluster].values)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_clust['Cluster'] = kmeans.fit_predict(caracteristiques_centrees).astype(str)
            
            acp_2d = PCA(n_components=2).fit_transform(caracteristiques_centrees)
            df_clust['CP1'] = acp_2d[:, 0]
            df_clust['CP2'] = acp_2d[:, 1]
            
            fig_cluster = px.scatter(df_clust, x='CP1', y='CP2', color='Cluster', hover_data=['ID_Parcelle', 'Rendement_t_ha'], title=f"Zones agricoles (k={n_clusters})")
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            st.subheader("Profils des zones")
            profil = df_clust.groupby('Cluster')[['Temperature_C', 'Humidite_Pct', 'pH_Sol', 'Rendement_t_ha']].mean()
            st.dataframe(profil.style.format("{:.1f}"), use_container_width=True)
            
            fig_box = px.box(df_clust, x='Cluster', y='Rendement_t_ha', color='Cluster', title="Rendement par zone")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Ajoutez plus de donnees pour le clustering.")