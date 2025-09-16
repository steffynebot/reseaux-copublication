import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

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
st.markdown(f"<h1 style='color:{PRIMARY_COLOR}'>Copublications d'auteurs Inria (Sophia & Bordeaux) avec le reste du monde</h1>", unsafe_allow_html=True)

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

    # KPI par centre
    if not df_filtered.empty:
        pubs_centre = df_filtered.groupby(centre_col)[hal_col].nunique().reset_index()
        st.subheader("📍 Publications par centre")
        cols = st.columns(len(pubs_centre))
        for i, row in pubs_centre.iterrows():
            cols[i].metric(row[centre_col], row[hal_col])

    # Graphique publications par année
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

    # WordCloud
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
# Onglet 3 : Carte interactive Monde
# -------------------
with tab3:
    st.header("Carte copublications")
    if st.button("Générer la carte"):
        df_map = df_filtered.dropna(subset=["Latitude", "Longitude"])
        if df_map.empty:
            st.warning("Aucune donnée valide pour tracer la carte.")
        else:
            inria_lat, inria_lon = 43.619, 7.071
            pubs_villes = df_map.groupby([ville_col]).agg({
                hal_col: "count", "Latitude": "first", "Longitude": "first"
            }).reset_index()
            max_pub = pubs_villes[hal_col].max()
            fig_map = go.Figure()

            for _, row in pubs_villes.iterrows():
                fig_map.add_trace(go.Scattermapbox(
                    lon=[row["Longitude"]], lat=[row["Latitude"]], mode="markers",
                    marker=go.scattermapbox.Marker(
                        size=8 + (row[hal_col] / max_pub) * 25,
                        color=ACCENT_COLOR, opacity=0.7
                    ),
                    text=f"{row[ville_col]} : {row[hal_col]} pubs",
                    hoverinfo="text"
                ))

            arcs = df_map.groupby([ville_col]).agg({
                "Latitude": "first", "Longitude": "first", hal_col: "count"
            }).reset_index().head(100)
            max_arc = arcs[hal_col].max()

            for _, row in arcs.iterrows():
                lon0, lat0 = inria_lon, inria_lat
                lon1, lat1 = row["Longitude"], row["Latitude"]
                t = np.linspace(0, 1, 6)
                lon_curve = lon0 * (1 - t) + lon1 * t
                lat_curve = lat0 * (1 - t) + lat1 * t + 0.3 * np.sin(np.pi * t)
                fig_map.add_trace(go.Scattermapbox(
                    lon=lon_curve, lat=lat_curve, mode="lines",
                    line=dict(width=0.5 + (row[hal_col] / max_arc) * 4,
                              color=f"rgba(30,144,255,{0.3 + 0.7 * (row[hal_col] / max_arc)})"),
                    opacity=0.6, hoverinfo="text",
                    text=f"Inria ↔ {row[ville_col]} : {row[hal_col]} pubs",
                    showlegend=False
                ))

            fig_map.add_trace(go.Scattermapbox(
                lon=[inria_lon], lat=[inria_lat],
                mode="markers", marker=dict(size=15, color=PRIMARY_COLOR, symbol="star"),
                text="Inria Sophia Antipolis", hoverinfo="text"
            ))

            fig_map.update_layout(
                mapbox=dict(style="carto-positron" if not is_dark else "carto-darkmatter",
                            center=dict(lat=42.5, lon=12.5), zoom=5),
                margin=dict(l=0, r=0, t=50, b=0),
                title="TOP 10 des villes copubliantes",
                plot_bgcolor=BACKGROUND_COLOR,
                paper_bgcolor=BACKGROUND_COLOR
            )
            st.plotly_chart(fig_map, use_container_width=True)

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

