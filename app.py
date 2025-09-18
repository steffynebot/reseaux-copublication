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
# Sidebar filtres
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
# Onglet 1 : KPI et dashboard moderne
# -------------------
with tab1:
    st.header("KPI et Dashboard")

    # Calculs de base
    pubs_year = compute_yearly(df_filtered)
    total_pubs = pubs_year[hal_col].sum()
    total_villes = df_filtered[ville_col].nunique()
    total_auteurs_inria = df_filtered[auteurs_fr_col].nunique()
    total_auteurs_copub = df_filtered[auteurs_copub_col].nunique()

    # Delta pour Publications
    delta_pubs = pubs_year[hal_col].iloc[-1] - pubs_year[hal_col].iloc[-2] if len(pubs_year) > 1 else 0

    # ---------------- KPI ----------------
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Publications", total_pubs, delta=f"{delta_pubs}" if delta_pubs != 0 else None)
    kpi_cols[1].metric("Villes", total_villes)
    kpi_cols[2].metric("Auteurs Inria", total_auteurs_inria)
    kpi_cols[3].metric("Auteurs copubliants", total_auteurs_copub)

    # Publications par centre
    if not df_filtered.empty:
        pubs_centre = df_filtered.groupby(centre_col)[hal_col].nunique().reset_index()
        st.subheader("Publications par centre")
        centre_cols = st.columns(len(pubs_centre))
        for i, row in pubs_centre.iterrows():
            centre_cols[i].metric(row[centre_col], row[hal_col])

    # ---------------- Graphiques ----------------
    st.subheader("Visualisations")
    graph_col1, graph_col2 = st.columns(2)

    # Publications par année
    fig_year = px.bar(
        pubs_year,
        x=annee_col,
        y=hal_col,
        title="Publications par année",
        color=hal_col,
        color_continuous_scale=px.colors.sequential.Plasma,
        text=hal_col
    )
    fig_year.update_traces(marker_line_color='black', marker_line_width=1.5)
    fig_year.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    graph_col1.plotly_chart(fig_year, use_container_width=True)

    # Top villes
    top_villes = compute_top(df_filtered, ville_col)
    fig_villes = go.Figure(
        data=[go.Pie(
            labels=top_villes.index,
            values=top_villes.values,
            hole=0.4,
            marker_colors=px.colors.qualitative.Pastel,
            pull=[0.05]*len(top_villes),
            textinfo='label+percent'
        )]
    )
    fig_villes.update_layout(title="TOP 10 des villes copubliantes")
    graph_col2.plotly_chart(fig_villes, use_container_width=True)

    # Top organismes
    top_orgs = compute_top(df_filtered, org_col)
    fig_orgs = go.Figure(
        data=[go.Pie(
            labels=top_orgs.index,
            values=top_orgs.values,
            hole=0.4,
            marker_colors=px.colors.qualitative.Set3,
            pull=[0.05]*len(top_orgs),
            textinfo='label+percent'
        )]
    )
    fig_orgs.update_layout(title="TOP 10 des organismes copubliants")
    st.plotly_chart(fig_orgs, use_container_width=True)

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

        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=0.5, color=NEUTRAL_COLOR),
                                hoverinfo="none", mode="lines")
        node_x, node_y, node_text, node_color = [], [], [], []
        color_map = {"Inria": PRIMARY_COLOR, "Copubliant": SECONDARY_COLOR, "Ville": ACCENT_COLOR}
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append(color_map.get(G.nodes[node]["type"], NEUTRAL_COLOR))

        node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                text=node_text, hoverinfo="text",
                                marker=dict(color=node_color, size=14, line_width=2))

        fig_net = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(title="TOP 10 des copublications par auteurs",
                                             showlegend=False, hovermode="closest",
                                             plot_bgcolor=BACKGROUND_COLOR,
                                             paper_bgcolor=BACKGROUND_COLOR))
        st.plotly_chart(fig_net, use_container_width=True)

import pydeck as pdk

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
