import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import math
import pydeck as pdk

# -------------------
# Page config
# -------------------
st.set_page_config(page_title="Copublications Inria (centres Sophia et Bordeaux)", layout="wide")

# -------------------
# Détection du thème actuel
# -------------------
theme = st.get_option("theme.base")  # 'light' ou 'dark'
is_dark = theme == "dark"

# -------------------
# Couleurs selon le mode
# -------------------
if is_dark:
    PRIMARY_COLOR = "#83c9ff"
    SECONDARY_COLOR = "#ffabab"
    ACCENT_COLOR = "#7defa1"
    NEUTRAL_COLOR = "#d5dae5"
    BACKGROUND_COLOR = "#004280"
    SIDEBAR_COLOR = "#0068c9"
else:
    PRIMARY_COLOR = "#0484fc"
    SECONDARY_COLOR = "#faa48a"
    ACCENT_COLOR = "#4cada3"
    NEUTRAL_COLOR = "#9ebfd2"
    BACKGROUND_COLOR = "#e4f5ff"
    SIDEBAR_COLOR = "#c7ebff"

# -------------------
# Load data
# -------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Copubliants_par_auteur_Inria_concat.xlsx")
    df.columns = [str(c).strip().replace("\xa0", "").replace(" ", "_") for c in df.columns]
    return df

df = load_data()
if df.empty:
    st.error("Aucune donnée trouvée.")
    st.stop()

# Colonnes
hal_col, auteurs_fr_col, auteurs_copub_col = "HalID", "Auteurs_FR", "Auteurs_copubliants"
ville_col, org_col, annee_col, equipe_col, centre_col = "Ville", "Organisme_copubliant", "Année", "Equipe", "Centre"
# -------------------
# Sidebar filtres améliorée
# -------------------
with st.sidebar:
    # Conteneur principal sans fond bleu
    st.markdown("<div style='padding:10px;border-radius:0.5rem;'>", unsafe_allow_html=True)

    # Texte "proposé par le groupe DATALAKE" en background léger
    st.markdown("""
        <div style='background-color:#f0f0f0;padding:10px;border-radius:0.5rem;text-align:center;margin-bottom:10px;font-size:12px;color:#555;'>
            Proposé par le groupe <b>DATALAKE</b> : Kumar Guha, Daniel Da Silva et Andréa NEBOT
        </div>
    """, unsafe_allow_html=True)

    # Logo
    try:
        st.image("logo.png", use_container_width=True)  # <--- corrigé ici
    except:
        st.markdown("**Logo manquant**")

    # Texte "DATALAKE" sous le logo
    st.markdown("""
        <h3 style='text-align:center;margin-top:5px;color:#111;'>DATALAKE</h3>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Filtres")

    centres = st.multiselect("Centre", sorted(df[centre_col].dropna().unique()))
    villes = st.selectbox("Ville", ["Toutes"] + sorted(df[ville_col].dropna().unique()))
    organismes = st.multiselect("Organismes copubliants", sorted(df[org_col].dropna().unique()))
    annees = st.multiselect("Années", sorted(df[annee_col].dropna().unique()))
    equipes = st.multiselect("Équipes", sorted(df[equipe_col].dropna().unique()))

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------
# Filtrage
# -------------------
df_filtered = df.copy()
if centres:
    df_filtered = df_filtered[df_filtered[centre_col].isin(centres)]
if villes != "Toutes":
    df_filtered = df_filtered[df_filtered[ville_col] == villes]
if organismes:
    df_filtered = df_filtered[df_filtered[org_col].isin(organismes)]
if annees:
    df_filtered = df_filtered[df_filtered[annee_col].isin(annees)]
if equipes:
    df_filtered = df_filtered[df_filtered[equipe_col].isin(equipes)]

# -------------------
# Fonctions utiles
# -------------------
@st.cache_data(ttl=300)
def compute_yearly(df):
    return df.groupby(annee_col)[hal_col].nunique().reset_index()

@st.cache_data(ttl=300)
def compute_top(df, col, n=10):
    return df[col].value_counts().nlargest(n)

@st.cache_data(ttl=300)
def build_graph(df, max_nodes=200):
    G = nx.Graph()
    subset = df.head(max_nodes)
    for _, row in subset.dropna(subset=[auteurs_fr_col, auteurs_copub_col, ville_col]).iterrows():
        G.add_node(row[auteurs_fr_col], type="Inria")
        G.add_node(row[auteurs_copub_col], type="Copubliant")
        G.add_node(row[ville_col], type="Ville")
        G.add_edge(row[auteurs_fr_col], row[auteurs_copub_col])
        G.add_edge(row[auteurs_copub_col], row[ville_col])
    pos = nx.spring_layout(G, k=0.3, iterations=10, seed=42)
    return G, pos

@st.cache_data
def make_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white" if not is_dark else "#004280",
                   colormap="tab10").generate(text)
    return wc

# -------------------
# Titre principal
# -------------------
st.markdown(f"<h1 style='color:{PRIMARY_COLOR}'>Copublications d'auteurs Inria (Sophia & Bordeaux)</h1>", unsafe_allow_html=True)

# -------------------
# Tabs
# -------------------
tab1, tab2, tab3, tab4 = st.tabs(["Visualisation générale", "Réseau copublication", "Carte du monde", "Contact"])
# -------------------
# Onglet 1 : Dashboard Ultra-Pro corrigé
# -------------------
with tab1:
    st.markdown("<h2 style='text-align:center;'>KPI et Dashboard</h2>", unsafe_allow_html=True)

    # ---------------- Calculs de base ----------------
    pubs_year = compute_yearly(df_filtered)
    total_pubs = pubs_year[hal_col].sum()
    total_villes = df_filtered[ville_col].nunique()
    total_auteurs_inria = df_filtered[auteurs_fr_col].nunique()
    total_auteurs_copub = df_filtered[auteurs_copub_col].nunique()

    # Publications par centres spécifiques (HalID uniques)
    pubs_par_centre = df_filtered.groupby(centre_col)[hal_col].nunique()
    pubs_bordeaux = df_filtered[df_filtered[ville_col] == "Bordeaux"][hal_col].nunique()
    pubs_sophia = df_filtered[df_filtered[ville_col] == "Sophia"][hal_col].nunique()
    
    delta_pubs = pubs_year[hal_col].iloc[-1] - pubs_year[hal_col].iloc[-2] if len(pubs_year) > 1 else 0

    # ---------------- KPI ovales ----------------
    kpi_style = """
    <div style="
        background: #e0f2f1; 
        border-radius: 25px; 
        padding: 20px; 
        text-align: center; 
        box-shadow: 3px 3px 10px rgba(0,0,0,0.15); 
        font-size: 20px;
        font-weight: bold;
        color:#111;
        margin:10px;">
        {title}<br><span style='font-size:28px'>{value}</span>{delta}
    </div>
    """

    # Création des KPI
    kpi_cols = st.columns(7)
    kpi_cols[0].markdown(
        kpi_style.format(
            title="Publications",
            value=total_pubs,
            delta=f"<br><span style='color:green'>{delta_pubs}</span>" if delta_pubs>=0 else f"<br><span style='color:red'>{delta_pubs}</span>"
        ),
        unsafe_allow_html=True
    )
    kpi_cols[1].markdown(kpi_style.format(title="Villes", value=total_villes, delta=""), unsafe_allow_html=True)
    kpi_cols[2].markdown(kpi_style.format(title="Auteurs Inria", value=total_auteurs_inria, delta=""), unsafe_allow_html=True)
    kpi_cols[3].markdown(kpi_style.format(title="Auteurs copubliants", value=total_auteurs_copub, delta=""), unsafe_allow_html=True)
    kpi_cols[4].markdown(kpi_style.format(title="Publications par centre", value=pubs_par_centre.sum(), delta=""), unsafe_allow_html=True)
    kpi_cols[5].markdown(kpi_style.format(title="Bordeaux", value=pubs_bordeaux, delta=""), unsafe_allow_html=True)
    kpi_cols[6].markdown(kpi_style.format(title="Sophia", value=pubs_sophia, delta=""), unsafe_allow_html=True)

    # ---------------- Graphique Publications par année ----------------
    st.subheader("Publications par année")
    fig_year = px.bar(
        pubs_year,
        x=annee_col,
        y=hal_col,
        color=hal_col,
        color_continuous_scale=px.colors.sequential.Teal,
        text=hal_col,
    )
    fig_year.update_traces(marker_line_color='black', marker_line_width=1.5, hovertemplate='%{x}: %{y}')
    fig_year.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_x=0.5,
        xaxis_title='Année',
        yaxis_title='Nombre de publications'
    )
    st.plotly_chart(fig_year, use_container_width=True)

    # ---------------- Graphiques TOP côte à côte ----------------
    st.subheader("TOP 10")
    top_villes = compute_top(df_filtered, ville_col)
    top_orgs = compute_top(df_filtered, org_col)
    col1, col2 = st.columns(2)

    # Pie chart TOP villes
    fig_villes = go.Figure(
        data=[go.Pie(
            labels=top_villes.index,
            values=top_villes.values,
            hole=0.4,
            marker_colors=px.colors.sequential.Teal[:len(top_villes)],
            pull=[0.05]*len(top_villes),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )]
    )
    fig_villes.update_layout(title="Villes copubliantes", title_x=0.5)
    col1.plotly_chart(fig_villes, use_container_width=True)

    # Pie chart TOP organismes
    fig_orgs = go.Figure(
        data=[go.Pie(
            labels=top_orgs.index,
            values=top_orgs.values,
            hole=0.4,
            marker_colors=px.colors.sequential.Teal[:len(top_orgs)],
            pull=[0.05]*len(top_orgs),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )]
    )
    fig_orgs.update_layout(title="Organismes copubliants", title_x=0.5)
    col2.plotly_chart(fig_orgs, use_container_width=True)

    # ---------------- WordCloud ----------------
    if "Mots-cles" in df_filtered.columns:
        if st.button("Générer le WordCloud"):
            text = " ".join(df_filtered["Mots-cles"].dropna().astype(str))
            if text:
                wc = WordCloud(width=800, height=400, background_color='white').generate(text)
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)

import dash
from dash import dcc, html
import dash_cytoscape as cyto
import pandas as pd

# Chargement des données
df_filtered = pd.read_csv("ton_fichier.csv")

# Préparation des éléments pour Cytoscape
elements = []
for _, row in df_filtered.iterrows():
    centre = row["Centre"]
    equipe = row["Equipe"]
    auteur_fr = row["Auteurs_FR"]
    auteur_cp = row["Auteurs_copubliants"]
    pays = row["Pays"]
    ville = row["Ville"]
    
    # Ajout des nœuds
    elements.append({"data": {"id": centre, "label": centre, "type": "Centre"}})
    elements.append({"data": {"id": equipe, "label": equipe, "type": "Equipe"}})
    elements.append({"data": {"id": auteur_fr, "label": auteur_fr, "type": "Auteur_FR"}})
    elements.append({"data": {"id": auteur_cp, "label": auteur_cp, "type": "Auteur_CP"}})
    elements.append({"data": {"id": pays, "label": pays, "type": "Pays"}})
    elements.append({"data": {"id": ville, "label": ville, "type": "Ville"}})
    
    # Ajout des arêtes
    elements.append({"data": {"source": centre, "target": equipe}})
    elements.append({"data": {"source": equipe, "target": auteur_fr}})
    elements.append({"data": {"source": auteur_fr, "target": auteur_cp}})
    elements.append({"data": {"source": auteur_cp, "target": pays}})
    elements.append({"data": {"source": pays, "target": ville}})

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Configuration de la mise en page de l'application
app.layout = html.Div([
    html.H1("Réseau de Copublication"),
    cyto.Cytoscape(
        id='cytoscape',
        elements=elements,
        layout={'name': 'breadthfirst', 'roots': '[id = "Centre"]'},
        style={'width': '100%', 'height': '600px'},
        stylesheet=[
            {'selector': 'node[type = "Centre"]', 'style': {'background-color': '#1f77b4', 'label': 'data(label)'}},
            {'selector': 'node[type = "Equipe"]', 'style': {'background-color': '#ff7f0e', 'label': 'data(label)'}},
            {'selector': 'node[type = "Auteur_FR"]', 'style': {'background-color': '#2ca02c', 'label': 'data(label)'}},
            {'selector': 'node[type = "Auteur_CP"]', 'style': {'background-color': '#d62728', 'label': 'data(label)'}},
            {'selector': 'node[type = "Pays"]', 'style': {'background-color': '#9467bd', 'label': 'data(label)'}},
            {'selector': 'node[type = "Ville"]', 'style': {'background-color': '#8c564b', 'label': 'data(label)'}},
            {'selector': 'edge', 'style': {'width': 2, 'line-color': '#888'}}
        ]
    )
])

# Exécution de l'application
if __name__ == '__main__':
    app.run_server(debug=True)


# -------------------
# Onglet 3 : Carte interactive Heatmap
# -------------------
with tab3:
    st.header("Carte des copublications")
    if st.button("Générer la carte"):
        df_map = df_filtered.dropna(subset=["Latitude", "Longitude"])
        if df_map.empty:
            st.warning("Aucune donnée valide pour tracer la carte.")
        else:
            # Définir les centres Inria
            inria_centers = [
                {"name": "Bordeaux", "lat": 44.833328, "lon": -0.56667, "color": [255, 0, 0]},
                {"name": "Sophia", "lat": 43.6200, "lon": 7.0500, "color": [0, 0, 255]}
            ]
            if centres:
                inria_centers = [c for c in inria_centers if c["name"].lower() in [cc.lower() for cc in centres]]

            # HeatmapLayer pour la densité des copublications
            heatmap_df = pd.DataFrame({
                "lon": df_map["Longitude"],
                "lat": df_map["Latitude"]
            })
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                heatmap_df,
                get_position=["lon", "lat"],
                get_weight=1,
                radius_pixels=25,
                opacity=0.6,
                threshold=0.03
            )

            # ScatterplotLayer pour les centres Inria
            centers_df = pd.DataFrame({
                "lon": [c["lon"] for c in inria_centers],
                "lat": [c["lat"] for c in inria_centers],
                "name": [c["name"] for c in inria_centers],
                "color": [c["color"] for c in inria_centers]
            })
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                centers_df,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius=15000,
                pickable=True,
            )

            # Vue centrée et ajustée
            view_state = pdk.ViewState(
                latitude=df_map["Latitude"].mean(),
                longitude=df_map["Longitude"].mean(),
                zoom=5,
                pitch=45,
                bearing=0
            )

            # Créer la carte
            deck = pdk.Deck(
                layers=[heatmap_layer, scatter_layer],
                initial_view_state=view_state,
                map_style=pdk.map_styles.CARTO_DARK,
                tooltip={"text": "{name}"}
            )

            # Afficher la carte
            st.pydeck_chart(deck)

# -------------------
# Onglet 4 : Contact
# -------------------
with tab4:
    st.header("À propos de nous")
    st.markdown("""
    Le groupe **Datalake**, créé en 2022, travaille à rendre possible le croisement de données entre **HAL** et divers référentiels et sources externes ou internes,
    de développer des outils et méthodes d’analyse et de prospection pour permettre à différents acteurs décisionnaires (**ADS, DPE, etc.**) ou scientifiques
    de répondre à leurs préoccupations du moment.  
    Il est constitué de **6 membres** aux profils de data scientistes, développeurs et documentalistes experts.
    """)
    st.markdown("---")
    st.header("📬 Formulaire de contact")
    with st.form("contact_form", clear_on_submit=True):
        nom = st.text_input("Votre nom")
        email = st.text_input("Votre email")
        message = st.text_area("Votre message")
        submitted = st.form_submit_button("Envoyer")
        if submitted:
            if not nom or not email or not message:
                st.error("⚠️ Merci de remplir tous les champs.")
            else:
                st.success(f"Merci {nom} ! Votre message a bien été envoyé ✅")
