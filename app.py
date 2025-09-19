import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
    BACKGROUND_COLOR = "#004280"
else:
    PRIMARY_COLOR = "#0484fc"
    SECONDARY_COLOR = "#faa48a"
    ACCENT_COLOR = "#4cada3"
    BACKGROUND_COLOR = "#e4f5ff"

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
ville_col, org_col, annee_col, equipe_col, centre_col, pays_col = "Ville", "Organisme_copubliant", "Année", "Equipe", "Centre", "Pays"

# -------------------
# Sidebar filtres
# -------------------
with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.caption("Logo manquant")
        
    st.markdown("### DATALAKE")
    
    centres = st.multiselect("Centre", sorted(df[centre_col].dropna().unique()))
    pays = st.multiselect("Pays", sorted(df[pays_col].dropna().unique()))
    villes = st.selectbox("Ville", ["Toutes"] + sorted(df[ville_col].dropna().unique()))
    organismes = st.multiselect("Organismes copubliants", sorted(df[org_col].dropna().unique()))
    annees = st.multiselect("Années", sorted(df[annee_col].dropna().unique()))
    equipes = st.multiselect("Équipes", sorted(df[equipe_col].dropna().unique()))

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Texte en bas du contenu des filtres
    st.caption(
        "Proposé par le groupe **DATALAKE** : Kumar Guha, Daniel Da Silva et Andréa Nebot  \n"
        "à la demande de Luigi Liquori et Maria Kazolea"
    )
    
# -------------------
# Filtrage
# -------------------
df_filtered = df.copy()
if centres:
    df_filtered = df_filtered[df_filtered[centre_col].isin(centres)]
if pays:
    df_filtered = df_filtered[df_filtered[pays_col].isin(pays)]
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
        G.add_node(row[pays_col], type="Pays")
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
st.title("Copublications d'auteurs Inria (Sophia & Bordeaux)")

# -------------------
# Tabs
# -------------------
tab1, tab2, tab3, tab4 = st.tabs(["Visualisation générale", "Réseau copublication", "Carte du monde", "Contact"])

# -------------------
# Onglet 1 : Dashboard
# -------------------
# -------------------
# Onglet 1 : Dashboard
# -------------------
with tab1:
    st.subheader("Indicateurs clés")
    pubs_year = compute_yearly(df_filtered)
    total_pubs = pubs_year[hal_col].sum()
    total_villes = df_filtered[ville_col].nunique()
    total_pays = df_filtered[pays_col].nunique()
    total_auteurs_inria = df_filtered[auteurs_fr_col].nunique()
    total_auteurs_copub = df_filtered[auteurs_copub_col].nunique()
    pubs_par_centre = df_filtered.groupby(centre_col)[hal_col].nunique()
    pubs_bordeaux = df_filtered[df_filtered[ville_col] == "Bordeaux"][hal_col].nunique()
    pubs_sophia = df_filtered[df_filtered[ville_col] == "Sophia"][hal_col].nunique()

    kpi_data = [
        ("Publications", total_pubs),
        ("Pays", total_pays),
        ("Villes", total_villes),
        ("Auteurs Inria", total_auteurs_inria),
        ("Auteurs copubliants", total_auteurs_copub),
        ("Publications par centre", pubs_par_centre.sum()),
        ("Bordeaux", pubs_bordeaux),
        ("Sophia", pubs_sophia),
    ]

    cols = st.columns(len(kpi_data))
    for col, (label, value) in zip(cols, kpi_data):
        col.metric(label, int(value))


    st.markdown("---")
    st.subheader("Publications par année")
    fig_year = px.bar(
        pubs_year,
        x=annee_col,
        y=hal_col,
        color=hal_col,
        color_continuous_scale=px.colors.sequential.Teal,
        text=hal_col,
    )
    fig_year.update_traces(marker_line_width=1.5, hovertemplate='%{x}: %{y}')
    fig_year.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           title_x=0.5, xaxis_title='Année', yaxis_title='Nombre de publications')
    st.plotly_chart(fig_year, use_container_width=True)

    st.subheader("TOP 10")
    top_villes = compute_top(df_filtered, ville_col)
    top_pays = compute_top(df_filtered, pays_col)
    top_orgs = compute_top(df_filtered, org_col)
    col1, col2 = st.columns(2)

    # Pie chart TOP villes
    fig_villes = go.Figure(go.Pie(labels=top_villes.index, values=top_villes.values, hole=0.4,
                                  marker_colors=px.colors.sequential.Teal[:len(top_villes)],
                                  textinfo='label+percent'))
    fig_villes.update_layout(title="Villes copubliantes", title_x=0.5)
    col1.plotly_chart(fig_villes, use_container_width=True)

    # Pie chart TOP organismes
    fig_orgs = go.Figure(go.Pie(labels=top_orgs.index, values=top_orgs.values, hole=0.4,
                                marker_colors=px.colors.sequential.Teal[:len(top_orgs)],
                                textinfo='label+percent'))
    fig_orgs.update_layout(title="Organismes copubliants", title_x=0.5)
    col2.plotly_chart(fig_orgs, use_container_width=True)


    # Pie chart TOP pays
    fig_pays = go.Figure(go.Pie(labels=top_pays.index, values=top_pays.values, hole=0.4,
                                marker_colors=px.colors.sequential.Teal[:len(top_orgs)],
                                textinfo='label+percent'))
    fig_pays.update_layout(title="Pays", title_x=0.5)
    col2.plotly_chart(fig_pays, use_container_width=True)
    
    # WordCloud
    if "Mots-cles" in df_filtered.columns and st.button("Générer le WordCloud"):
        text = " ".join(df_filtered["Mots-cles"].dropna().astype(str))
        if text:
            wc = make_wordcloud(text)
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)

# -------------------
# Onglet 2 : Réseau copublication
# -------------------
with tab2:
    st.header("Réseau de copublication")
    if st.button("Générer le réseau"):
        max_nodes = 200
        subset = df_filtered.head(max_nodes)
        G = nx.Graph()
        for _, row in subset.dropna(subset=[auteurs_fr_col, auteurs_copub_col, ville_col]).iterrows():
            G.add_node(row[centre_col], type="Centre")
            G.add_node(row[equipe_col], type="Equipe")
            G.add_node(row[auteurs_fr_col], type="Auteur_FR")
            G.add_node(row[auteurs_copub_col], type="Auteur_CP")
            G.add_node(row["Pays"], type="Pays")
            G.add_node(row[ville_col], type="Ville")
            G.add_edges_from([
                (row[centre_col], row[equipe_col]),
                (row[equipe_col], row[auteurs_fr_col]),
                (row[auteurs_fr_col], row[auteurs_copub_col]),
                (row[auteurs_copub_col], row["Pays"]),
                (row["Pays"], row[ville_col])
            ])

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"),
                                hoverinfo="none", mode="lines", showlegend=False)

        color_map = {"Centre": "#1f77b4", "Equipe": "#ff7f0e", "Auteur_FR": "#2ca02c",
                     "Auteur_CP": "#d62728", "Pays": "#9467bd", "Ville": "#8c564b"}
        node_degree = dict(G.degree())
        node_traces = []
        for node_type, color in color_map.items():
            node_x, node_y, node_text, node_size = [], [], [], []
            for node in G.nodes():
                if G.nodes[node]["type"] == node_type:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"{node} ({node_type}) - {node_degree[node]} copubs")
                    node_size.append(10 + node_degree[node]*2)
            if node_x:
                node_traces.append(go.Scatter(x=node_x, y=node_y, mode="markers", name=node_type,
                                              hovertext=node_text, hoverinfo="text",
                                              marker=dict(color=color, size=node_size, line_width=2)))
        fig_net = go.Figure(data=[edge_trace]+node_traces,
                             layout=go.Layout(title="Réseau des copublications",
                                              showlegend=True, legend=dict(title="Type de nœud"),
                                              hovermode="closest", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"))
        st.plotly_chart(fig_net, use_container_width=True)

# -------------------
# Onglet 3 : Carte interactive
# -------------------
with tab3:
    st.header("Carte des copublications")
    if st.button("Générer la carte"):
        df_map = df_filtered.dropna(subset=["Latitude", "Longitude"])
        if df_map.empty:
            st.warning("Aucune donnée valide pour tracer la carte.")
        else:
            inria_centers = [
                {"name": "Bordeaux", "lat": 44.833328, "lon": -0.56667, "color": [255,0,0]},
                {"name": "Sophia", "lat": 43.6200, "lon": 7.0500, "color": [0,0,255]}
            ]
            if centres:
                inria_centers = [c for c in inria_centers if c["name"].lower() in [cc.lower() for cc in centres]]
            heatmap_df = pd.DataFrame({"lon": df_map["Longitude"], "lat": df_map["Latitude"]})
            heatmap_layer = pdk.Layer("HeatmapLayer", heatmap_df, get_position=["lon","lat"],
                                      get_weight=1, radius_pixels=25, opacity=0.6, threshold=0.03)
            centers_df = pd.DataFrame({"lon":[c["lon"] for c in inria_centers],
                                       "lat":[c["lat"] for c in inria_centers],
                                       "name":[c["name"] for c in inria_centers],
                                       "color":[c["color"] for c in inria_centers]})
            scatter_layer = pdk.Layer("ScatterplotLayer", centers_df, get_position=["lon","lat"],
                                      get_fill_color="color", get_radius=15000, pickable=True)
            view_state = pdk.ViewState(latitude=df_map["Latitude"].mean(),
                                       longitude=df_map["Longitude"].mean(),
                                       zoom=5, pitch=45, bearing=0)
            deck = pdk.Deck(layers=[heatmap_layer, scatter_layer],
                            initial_view_state=view_state,
                            map_style=pdk.map_styles.CARTO_DARK,
                            tooltip={"text":"{name}"})
            st.pydeck_chart(deck)

# -------------------
# Onglet 4 : Contact
# -------------------
with tab4:
    st.header("À propos de nous")
    st.markdown("""
    Le groupe **Datalake**, créé en 2022, travaille à rendre possible le croisement de données entre **HAL** et divers référentiels,
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
