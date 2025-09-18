import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pydeck as pdk

# -------------------
# Page config
# -------------------
st.set_page_config(page_title="Copublications Inria-Italie", layout="wide")

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
    df_sophia = pd.read_excel("Copubliants_par_auteur_Inria_Sophia_ville_final_long_lat.xlsx")
    df_bordeaux = pd.read_excel("Copubliants_par_auteur_Inria_Bordeaux_ville_final_long_lat.xlsx")
    df = pd.concat([df_sophia, df_bordeaux], ignore_index=True)
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
# Sidebar
# -------------------
with st.sidebar:
    st.markdown(f"<div style='background-color:{SIDEBAR_COLOR};padding:10px;border-radius:0.5rem'>", unsafe_allow_html=True)
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.markdown("**Logo manquant**")
    st.markdown("---")
    st.header("Filtres")

    centres = st.multiselect("Centre", sorted(df[centre_col].dropna().unique()))
    villes = st.selectbox("Ville", ["Toutes"] + sorted(df[ville_col].dropna().unique()))
    organismes = st.multiselect("Organismes copubliants", sorted(df[org_col].dropna().unique()))
    annees = st.multiselect("Années", sorted(df[annee_col].dropna().unique()))
    equipes = st.multiselect("Équipes", sorted(df[equipe_col].dropna().unique()))

    st.markdown("---")
    st.markdown(f"<p style='text-align:center;color:{PRIMARY_COLOR}'>Proposé par <b>Andréa NEBOT</b></p>", unsafe_allow_html=True)
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
# Fonctions
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
st.markdown(f"<h1 style='color:{PRIMARY_COLOR}'>Copublications d'auteurs Inria (Sophia & Bordeaux) </h1>", unsafe_allow_html=True)

# -------------------
# Tabs
# -------------------
tab1, tab2, tab3, tab4 = st.tabs(["Visualisation générale", "Réseau copublication", "Carte du monde", "Contact"])

# -------------------
# Onglet 1 : KPI et graphiques
# -------------------
with tab1:
    st.header("KPI et graphiques")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Publications (HalID uniques)", df_filtered[hal_col].nunique())
    col2.metric("Nombre de villes", df_filtered[ville_col].nunique())
    col3.metric("Auteurs Inria", df_filtered[auteurs_fr_col].nunique())
    col4.metric("Auteurs copubliants", df_filtered[auteurs_copub_col].nunique())

    if not df_filtered.empty:
        pubs_centre = df_filtered.groupby(centre_col)[hal_col].nunique().reset_index()
        st.subheader("📍 Publications par centre")
        cols = st.columns(len(pubs_centre))
        for i, row in pubs_centre.iterrows():
            cols[i].metric(row[centre_col], row[hal_col])

    pubs_year = compute_yearly(df_filtered)
    fig_year = px.bar(pubs_year, x=annee_col, y=hal_col, title="Publications par année",
                      color=hal_col, color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig_year, use_container_width=True)

    # Top 10 villes
    top_villes = compute_top(df_filtered, ville_col)
    fig_villes = go.Figure(data=[go.Pie(labels=top_villes.index, values=top_villes.values, hole=0.4)])
    fig_villes.update_traces(marker=dict(colors=[ACCENT_COLOR]*len(top_villes)))
    fig_villes.update_layout(title="TOP 10 des villes copubliantes")
    st.plotly_chart(fig_villes, use_container_width=True)

    # Top 10 organismes
    top_orgs = compute_top(df_filtered, org_col)
    fig_orgs = go.Figure(data=[go.Pie(labels=top_orgs.index, values=top_orgs.values, hole=0.4)])
    fig_orgs.update_traces(marker=dict(colors=[SECONDARY_COLOR]*len(top_orgs)))
    fig_orgs.update_layout(title="TOP 10 des organismes copubliants")
    st.plotly_chart(fig_orgs, use_container_width=True)

    if "Mots-cles" in df_filtered.columns:
        if st.button("Générer le WordCloud"):
            text = " ".join(df_filtered["Mots-cles"].dropna().astype(str))
            if text:
                wc = make_wordcloud(text)
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)

# -------------------
# Onglet 2 : Réseau interactif
# -------------------
with tab2:
    st.header("Réseau copublication")
    if st.button("Générer le réseau"):
        G, pos = build_graph(df_filtered)
        st.info(f"Réseau limité à {len(G.nodes)} nœuds.")

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=NEUTRAL_COLOR),
            hoverinfo="none",
            mode="lines"
        )

        node_x, node_y, node_text, node_color = [], [], [], []
        color_map = {"Inria": PRIMARY_COLOR, "Copubliant": SECONDARY_COLOR, "Ville": ACCENT_COLOR}
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append(color_map.get(G.nodes[node]["type"], NEUTRAL_COLOR))

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=node_text, hoverinfo="text",
            marker=dict(color=node_color, size=14, line_width=2)
        )

        fig_net = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(title="TOP 10 des copublications par auteurs",
                                             showlegend=False, hovermode="closest",
                                             plot_bgcolor=BACKGROUND_COLOR,
                                             paper_bgcolor=BACKGROUND_COLOR))
        st.plotly_chart(fig_net, use_container_width=True)

# -------------------
# Sidebar : choix du mode arcs
# -------------------
with st.sidebar:
    arc_mode = st.selectbox(
        "Mode d'affichage des arcs",
        ["Couleur par centre (optimisé)", "Multicolors (visuel)"]
    )

import numpy as np

def make_arc(lat1, lon1, lat2, lon2, n_points=20, curve_height=0.5):
    """
    Crée une courbe arrondie entre deux points (lat1, lon1) -> (lat2, lon2).
    n_points : nombre de points pour lisser la courbe
    curve_height : facteur d'arrondi (0 = ligne droite, >0 = plus arrondi)
    """
    lats = np.linspace(lat1, lat2, n_points)
    
import math

def make_arc(lat1, lon1, lat2, lon2, n_points=30, curve_height=0.5):
    """
    Retourne (lats, lons) d'un arc arrondi reliant (lat1, lon1) à (lat2, lon2).
    curve_height : proportion de la "hauteur" de la courbe (0 = droite, >0 = plus arrondi).
    """
    # points linéaires de base
    t = np.linspace(0, 1, n_points)
    lats = lat1 + (lat2 - lat1) * t
    lons = lon1 + (lon2 - lon1) * t

    # vecteur direction
    dx = lon2 - lon1
    dy = lat2 - lat1

    # vecteur perpendiculaire normalisé
    perp_x = -dy
    perp_y = dx
    norm = math.hypot(perp_x, perp_y)
    if norm == 0:
        perp_x, perp_y = 0, 0
    else:
        perp_x /= norm
        perp_y /= norm

    # amplitude de la courbe proportionnelle à la distance
    dist = math.hypot(dx, dy)
    amp = curve_height * dist

    # appliquer courbure suivant une sinusoidale centrée
    curve = np.sin(np.pi * t) * amp
    lats = lats + perp_y * curve
    lons = lons + perp_x * curve

    return lats.tolist(), lons.tolist()


# -------------------
# Onglet 3 : Carte interactive avec arcs arrondis
# -------------------
with tab3:
    st.header("Carte copublications Italie")
    if st.button("Générer la carte"):
        df_map = df_filtered.dropna(subset=["Latitude", "Longitude"])
        if df_map.empty:
            st.warning("Aucune donnée valide pour tracer la carte.")
        else:
            # Coordonnées des centres
            centers = [
                {"name": "Bordeaux", "lat": 44.833328, "lon": -0.56667, "color": "#1f77b4"},
                {"name": "Sophia", "lat": 43.6200, "lon": 7.0500, "color": "#d62728"}
            ]
            if centres:
                centers = [c for c in centers if c["name"].lower() in [cc.lower() for cc in centres]]

            # Points des copubliants
            fig = px.scatter_mapbox(
                df_map,
                lat="Latitude",
                lon="Longitude",
                hover_name=ville_col,
                hover_data={org_col: True, annee_col: True},
                color=annee_col,
                size_max=15,
                zoom=4,
                height=600
            )

            # Ajouter les centres Inria
            for center in centers:
                fig.add_scattermapbox(
                    lat=[center["lat"]],
                    lon=[center["lon"]],
                    mode="markers+text",
                    marker=dict(size=18, color=center["color"], symbol="star"),
                    text=[center["name"]],
                    textposition="top right",
                    name=f"Centre {center['name']}"
                )

            # Arcs arrondis avec hover personnalisé
            for center in centers:
                for _, row in df_map.iterrows():
                    try:
                        lat2 = float(row["Latitude"])
                        lon2 = float(row["Longitude"])
                        lat1 = float(center["lat"])
                        lon1 = float(center["lon"])
                    except Exception:
                        continue  # ignorer si valeurs invalides

                    arc_lats, arc_lons = make_arc(lat1, lon1, lat2, lon2,
                                                  n_points=8, curve_height=0.6)

                    hover_text = (
                        f"<b>Auteur copubliant :</b> {row.get(auteurs_copub_col, '')}<br>"
                        f"<b>Pays :</b> {row.get('Pays', '')}<br>"
                        f"<b>Ville :</b> {row.get('Ville', '')}"
                    )

                    fig.add_scattermapbox(
                        lat=arc_lats,
                        lon=arc_lons,
                        mode="lines",
                        line=dict(width=1.5, color=center["color"]),
                        opacity=0.6,
                        hoverinfo="text",
                        text=[hover_text] * len(arc_lats),
                        showlegend=False
                    )

            # Mise en forme
            fig.update_layout(
                mapbox_style="carto-positron",
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                legend=dict(y=0.99, x=0.01)
            )

            st.plotly_chart(fig, use_container_width=True)

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
