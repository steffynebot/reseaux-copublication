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
from collections import Counter, defaultdict

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
# Palette UI/UX tendance (exemple)
# Remplace ces valeurs par celles que tu choisis sur Adobe
# -------------------
PALETTE = {
    "dark_blue": "#264653",    # couleur neutre sombre / fond
    "teal_blue": "#2a9d8f",    # primaire
    "sand_yellow": "#e9c46a",  # accent clair
    "orange_warm": "#f4a261",  # secondaire
    "coral_red": "#e76f51",    # accent fort
    "light_bg": "#f8f4ec",      # fond clair
    "dark_bg": "#1a1a1a",       # fond sombre
    "white": "#ffffff",
    "light_neutral": "#e0e1dd",
}

# -------------------
# Couleurs selon le mode
# -------------------
if is_dark:
    PRIMARY_COLOR = PALETTE["teal_blue"]
    SECONDARY_COLOR = PALETTE["orange_warm"]
    ACCENT_COLOR = PALETTE["sand_yellow"]
    NEUTRAL_COLOR = PALETTE["dark_blue"]
    BACKGROUND_COLOR = PALETTE["dark_bg"]
    SIDEBAR_COLOR = PALETTE["dark_blue"]
    TEXT_COLOR = PALETTE["white"]
else:
    PRIMARY_COLOR = PALETTE["dark_blue"]
    SECONDARY_COLOR = PALETTE["coral_red"]
    ACCENT_COLOR = PALETTE["sand_yellow"]
    NEUTRAL_COLOR = PALETTE["white"]
    BACKGROUND_COLOR = PALETTE["light_bg"]
    SIDEBAR_COLOR = PALETTE["light_neutral"]
    TEXT_COLOR = PALETTE["dark_blue"]

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
# Sidebar filtres améliorée avec couleur
# -------------------
with st.sidebar:
    st.markdown(
        f"<div style='background-color:{SIDEBAR_COLOR};padding:10px;border-radius:0.5rem;'>",
        unsafe_allow_html=True
    )

    st.markdown(f"""
        <div style='background-color:{ACCENT_COLOR};padding:10px;border-radius:0.5rem;
                    text-align:center;margin-bottom:10px;
                    font-size:12px;color:{TEXT_COLOR};'>
            Proposé par le groupe <b>DATALAKE</b> : Kumar Guha, Daniel Da Silva et Andréa NEBOT
        </div>
    """, unsafe_allow_html=True)

    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.markdown(f"<p style='color:{TEXT_COLOR};'>Logo manquant</p>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='text-align:center;margin-top:5px;color:{PRIMARY_COLOR};'>DATALAKE</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("Filtres", anchor=None)
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
        G.add_edge(row[auteurs_fr_col], row[auteurs_cop_col])
        G.add_edge(row[auteurs_cop_col], row[ville_col])
    pos = nx.spring_layout(G, k=0.3, iterations=10, seed=42)
    return G, pos

@st.cache_data
def make_wordcloud(text):
    wc = WordCloud(
        width=800, height=400,
        background_color=BACKGROUND_COLOR if not is_dark else PALETTE["dark_blue"],
        colormap="tab10"
    ).generate(text)
    return wc

# -------------------
# Titre principal
# -------------------
st.markdown(
    f"<h1 style='color:{PRIMARY_COLOR}'>Copublications d'auteurs Inria (Sophia & Bordeaux)</h1>",
    unsafe_allow_html=True
)

# -------------------
# Tabs
# -------------------
tab1, tab2, tab3, tab4 = st.tabs(["Visualisation générale", "Réseau copublication", "Carte du monde", "Contact"])

# -------------------
# Onglet 1 : Dashboard avec KPI & graphes
# -------------------
with tab1:
    st.markdown(f"<h2 style='text-align:center;color:{SECONDARY_COLOR};'>KPI et Dashboard</h2>", unsafe_allow_html=True)

    # Calculs de base
    pubs_year = compute_yearly(df_filtered)
    total_pubs = pubs_year[hal_col].sum()
    total_villes = df_filtered[ville_col].nunique()
    total_auteurs_inria = df_filtered[auteurs_fr_col].nunique()
    total_auteurs_copub = df_filtered[auteurs_copub_col].nunique()
    pubs_par_centre = df_filtered.groupby(centre_col)[hal_col].nunique()
    pubs_bordeaux = df_filtered[df_filtered[ville_col] == "Bordeaux"][hal_col].nunique()
    pubs_sophia = df_filtered[df_filtered[ville_col] == "Sophia"][hal_col].nunique()
    delta_pubs = (
        pubs_year[hal_col].iloc[-1] - pubs_year[hal_col].iloc[-2]
        if len(pubs_year) > 1
        else 0
    )

    # KPI ovales
    kpi_style = f"""
    <div style="
        background:{NEUTRAL_COLOR};
        border-radius: 25px;
        padding: 20px;
        text-align: center;
        box-shadow: 4px 4px 15px rgba(0,0,0,0.2);
        font-size: 20px;
        font-weight: bold;
        color:{TEXT_COLOR};
        margin:10px;
    ">
        {{title}}<br><span style='font-size:28px;color:{PRIMARY_COLOR}'>{{value}}</span>{{delta}}
    </div>
    """

    kpi_cols = st.columns(7)
    kpi_cols[0].markdown(
        kpi_style.format(
            title="Publications",
            value=total_pubs,
            delta=f"<br><span style='color:{ACCENT_COLOR}'>+{delta_pubs}</span>"
            if delta_pubs >= 0
            else f"<br><span style='color:{PALETTE['coral_red']}'>{delta_pubs}</span>"
        ),
        unsafe_allow_html=True
    )
    kpi_cols[1].markdown(kpi_style.format(title="Villes", value=total_villes, delta=""), unsafe_allow_html=True)
    kpi_cols[2].markdown(kpi_style.format(title="Auteurs Inria", value=total_auteurs_inria, delta=""), unsafe_allow_html=True)
    kpi_cols[3].markdown(kpi_style.format(title="Auteurs copubliants", value=total_auteurs_copub, delta=""), unsafe_allow_html=True)
    kpi_cols[4].markdown(kpi_style.format(title="Publications par centre", value=pubs_par_centre.sum(), delta=""), unsafe_allow_html=True)
    kpi_cols[5].markdown(kpi_style.format(title="Bordeaux", value=pubs_bordeaux, delta=""), unsafe_allow_html=True)
    kpi_cols[6].markdown(kpi_style.format(title="Sophia", value=pubs_sophia, delta=""), unsafe_allow_html=True)

    # Graphique Publications par année
    st.subheader("Publications par année")
    fig_year = px.bar(
        pubs_year,
        x=annee_col,
        y=hal_col,
        text=hal_col,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig_year.update_traces(marker_line_color=NEUTRAL_COLOR, marker_line_width=1.2, hovertemplate='%{x}: %{y}')
    fig_year.update_layout(
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        title_x=0.5,
        xaxis_title='Année',
        yaxis_title='Nombre de publications'
    )
    st.plotly_chart(fig_year, use_container_width=True)

    # Graphiques TOP côte à côte
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
            marker_colors=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, PALETTE["orange_warm"], PALETTE["coral_red"]][: len(top_villes)],
            pull=[0.05]*len(top_villes),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )]
    )
    fig_villes.update_layout(
        title="Villes copubliantes",
        title_x=0.5,
        font=dict(color=TEXT_COLOR),
        plot_bgcolor=BACKGROUND_COLOR, paper_bgcolor=BACKGROUND_COLOR
    )
    col1.plotly_chart(fig_villes, use_container_width=True)

    # Pie chart TOP organismes
    fig_orgs = go.Figure(
        data=[go.Pie(
            labels=top_orgs.index,
            values=top_orgs.values,
            hole=0.4,
            marker_colors=[PRIMARY_COLOR, ACCENT_COLOR, SECONDARY_COLOR, PALETTE["orange_warm"], PALETTE["coral_red"]][: len(top_orgs)],
            pull=[0.05]*len(top_orgs),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )]
    )
    fig_orgs.update_layout(
        title="Organismes copubliants",
        title_x=0.5,
        font=dict(color=TEXT_COLOR),
        plot_bgcolor=BACKGROUND_COLOR, paper_bgcolor=BACKGROUND_COLOR
    )
    col2.plotly_chart(fig_orgs, use_container_width=True)

    # WordCloud
    if "Mots-cles" in df_filtered.columns:
        if st.button("Générer le WordCloud"):
            text = " ".join(df_filtered["Mots-cles"].dropna().astype(str))
            if text:
                wc = WordCloud(
                    width=800, height=400,
                    background_color=BACKGROUND_COLOR if not is_dark else PALETTE["dark_blue"],
                    colormap="tab10"
                ).generate(text)
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)

# -------------------
# Onglet 2 : Réseau
# -------------------
with tab2:
    st.markdown(f"<h2 style='color:{PRIMARY_COLOR}'>Réseau de copublication</h2>", unsafe_allow_html=True)
    if st.button("Générer le réseau"):
        max_nodes = 200
        subset = df_filtered.head(max_nodes)

        G = nx.Graph()
        for _, row in subset.dropna(subset=[auteurs_fr_col, auteurs_copub_col, ville_col]).iterrows():
            G.add_node(row[centre_col], type="Centre")
            G.add_node(row[equipe_col], type="Equipe")
            G.add_node(row[auteurs_fr_col], type="Auteur_FR")
            G.add_node(row[auteurs_copub_col], type="Auteur_CP")
            # autres noeuds si tu as
            # …

            # Arêtes
            G.add_edge(row[centre_col], row[equipe_col])
            G.add_edge(row[equipe_col], row[auteurs_fr_col])
            G.add_edge(row[auteurs_fr_col], row[auteurs_copub_col])

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=SECONDARY_COLOR),
            hoverinfo="none",
            mode="lines",
            showlegend=False
        )

        color_map = {
            "Centre": PRIMARY_COLOR,
            "Equipe": ACCENT_COLOR,
            "Auteur_FR": SECONDARY_COLOR,
            "Auteur_CP": PALETTE["coral_red"],
            # etc.
        }

        node_degree = dict(G.degree())

        node_traces = []
        for node_type, color in color_map.items():
            node_x, node_y, node_text, node_size = [], [], [], []
            for node in G.nodes():
                if G.nodes[node].get("type") == node_type:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"{node} ({node_type}) - {node_degree[node]} copubs")
                    node_size.append(10 + node_degree[node] * 2)
            if node_x:
                node_traces.append(
                    go.Scatter(
                        x=node_x, y=node_y,
                        mode="markers",
                        name=node_type,
                        hovertext=node_text,
                        hoverinfo="text",
                        marker=dict(color=color, size=node_size, line_width=1.5)
                    )
                )

        fig_net = go.Figure(
            data=[edge_trace] + node_traces,
            layout=go.Layout(
                title=dict(text="Réseau des copublications", x=0.5, font=dict(color=PRIMARY_COLOR)),
                showlegend=True,
                legend=dict(title="Type de nœud", font=dict(color=TEXT_COLOR)),
                hovermode="closest",
                plot_bgcolor=BACKGROUND_COLOR,
                paper_bgcolor=BACKGROUND_COLOR,
                font=dict(color=TEXT_COLOR)
            )
        )
        st.plotly_chart(fig_net, use_container_width=True)

# -------------------
# Onglet 3 : Carte interactive
# -------------------
with tab3:
    st.header(f"<span style='color:{SECONDARY_COLOR}'>Carte des copublications</span>", unsafe_allow_html=True)
    if st.button("Générer la carte"):
        df_map = df_filtered.dropna(subset=["Latitude", "Longitude"])
        if df_map.empty:
            st.warning("Aucune donnée valide pour tracer la carte.")
        else:
            inria_centers = [
                {"name": "Bordeaux", "lat": 44.833328, "lon": -0.56667, "color": [int(PALETTE['teal_blue'][1:3],16),
                                                                                     int(PALETTE['teal_blue'][3:5],16),
                                                                                     int(PALETTE['teal_blue'][5:7],16)]},
                {"name": "Sophia", "lat": 43.6200, "lon": 7.0500, "color": [int(PALETTE['coral_red'][1:3],16),
                                                                            int(PALETTE['coral_red'][3:5],16),
                                                                            int(PALETTE['coral_red'][5:7],16)]}
            ]
            if centres:
                inria_centers = [c for c in inria_centers if c["name"].lower() in [cc.lower() for cc in centres]]

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

            view_state = pdk.ViewState(
                latitude=float(df_map["Latitude"].mean()),
                longitude=float(df_map["Longitude"].mean()),
                zoom=5,
                pitch=45,
                bearing=0
            )

            deck = pdk.Deck(
                layers=[heatmap_layer, scatter_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v10" if is_dark else "mapbox://styles/mapbox/light-v10",
                tooltip={"text": "{name}"}
            )

            st.pydeck_chart(deck)

# -------------------
# Onglet 4 : Contact
# -------------------
with tab4:
    st.header(f"<span style='color:{PRIMARY_COLOR}'>À propos de nous</span>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='color:{TEXT_COLOR};'>
        Le groupe <b>Datalake</b>, créé en 2022, travaille à rendre possible le croisement de données entre <b>HAL</b> et divers référentiels et sources externes ou internes,
        de développer des outils et méthodes d’analyse et de prospection pour permettre à différents acteurs décisionnaires (<b>ADS, DPE, etc.</b>) ou scientifiques
        de répondre à leurs préoccupations du moment.  
        Il est constitué de <b>6 membres</b> aux profils de data scientistes, développeurs et documentalistes experts.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.header(f"<span style='color:{SECONDARY_COLOR}'>📬 Formulaire de contact</span>", unsafe_allow_html=True)
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

