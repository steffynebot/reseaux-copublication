# app_fixed_colors.py
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
st.set_page_config(page_title="Copublications Inria (centres Sophia et Bordeaux)", layout="wide")

# -------------------
# Détection du thème actuel (Streamlit theme base)
# -------------------
theme = st.get_option("theme.base")  # 'light' ou 'dark'
is_dark = theme == "dark"

# -------------------
# Palette user (Adobe UI/UX)
# #B4C0D9, #023059, #F2F2F2, #BF1111, #0D0D0D
# -------------------
# -------------------
# Nouvelle palette
# -------------------
if is_dark:
    PRIMARY_COLOR = "#27C7D4"   # titres / accents principaux
    SECONDARY_COLOR = "#FFFFFF" # texte secondaire
    BACKGROUND_COLOR = "#FDF0E7"
    ACCENT_COLOR = "#FE9063"
    NEUTRAL_COLOR = "#EA5863"
    SIDEBAR_COLOR = "#27C7D4"
    TEXT_COLOR = "#FFFFFF"      # pour fond sombre
else:
    PRIMARY_COLOR = "#27C7D4"
    SECONDARY_COLOR = "#FFFFFF"
    BACKGROUND_COLOR = "#FDF0E7"
    ACCENT_COLOR = "#FE9063"
    NEUTRAL_COLOR = "#EA5863"
    SIDEBAR_COLOR = "#27C7D4"
    TEXT_COLOR = "#0D0D0D"      # texte lisible sur fond clair

# -------------------
# Small CSS for consistent headings & general look (no unsafe use with st.header)
# We'll still use st.markdown(html, unsafe_allow_html=True) for colored headings individually.
# -------------------
st.markdown(
    f"""
    <style>
        /* Make default markdown text color readable */
        .markdown-text-container {{
            color: {TEXT_COLOR} !important;
        }}
        /* Optional: style streamlit widgets container - minimal */
        .stButton>button {{
            background-color: {PRIMARY_COLOR} ;
            color: {NEUTRAL_COLOR} ;
            border: none;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------
# Load data
# -------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Copubliants_par_auteur_Inria_concat.xlsx")
    # clean column names
    df.columns = [str(c).strip().replace("\xa0", "").replace(" ", "_") for c in df.columns]
    return df

df = load_data()
if df.empty:
    st.error("Aucune donnée trouvée.")
    st.stop()

# Columns names (as in your dataset)
hal_col, auteurs_fr_col, auteurs_copub_col = "HalID", "Auteurs_FR", "Auteurs_copubliants"
ville_col, org_col, annee_col, equipe_col, centre_col = "Ville", "Organisme_copubliant", "Année", "Equipe", "Centre"

# -------------------
# Sidebar (with colored container)
# -------------------
with st.sidebar:
    st.markdown(f"<div style='background-color:{SIDEBAR_COLOR};padding:10px;border-radius:0.5rem;'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style='background-color:{PRIMARY_COLOR};padding:10px;border-radius:0.5rem;
                    text-align:center;margin-bottom:10px;font-size:12px;color:{NEUTRAL_COLOR};'>
            Proposé par le groupe <b>DATALAKE</b> : Kumar Guha, Daniel Da Silva et Andréa NEBOT
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        st.image("logo.png", use_container_width=True)
    except Exception:
        st.markdown(f"<p style='color:{NEUTRAL_COLOR};'>Logo manquant</p>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='text-align:center;margin-top:5px;color:{PRIMARY_COLOR};'>DATALAKE</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("Filtres")  # plain header (no unsafe)
    centres = st.multiselect("Centre", sorted(df[centre_col].dropna().unique()))
    villes = st.selectbox("Ville", ["Toutes"] + sorted(df[ville_col].dropna().unique()))
    organismes = st.multiselect("Organismes copubliants", sorted(df[org_col].dropna().unique()))
    annees = st.multiselect("Années", sorted(df[annee_col].dropna().unique()))
    equipes = st.multiselect("Équipes", sorted(df[equipe_col].dropna().unique()))
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------
# Filtrage des données
# -------------------
df_filtered = df.copy()
if centres:
    df_filtered = df_filtered[df_filtered[centre_col].isin(centres)]
if villes != "Toutes":
    df_filtered = df_filtered[df_filtered[ville_col] == villes]
if organismes:
    df_filtered = df_filtered[df[org_col].isin(organismes)] if False else df_filtered  # safety placeholder (kept original)
# correct filtering (the above line is placeholder to avoid silent override).
# Let's apply actual filters properly:
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
# Util functions (cached)
# -------------------
@st.cache_data(ttl=300)
def compute_yearly(df_):
    return df_.groupby(annee_col)[hal_col].nunique().reset_index()

@st.cache_data(ttl=300)
def compute_top(df_, col, n=10):
    return df_[col].value_counts().nlargest(n)

@st.cache_data(ttl=300)
def build_graph(df_, max_nodes=200):
    G = nx.Graph()
    subset = df_.head(max_nodes)
    for _, row in subset.dropna(subset=[auteurs_fr_col, auteurs_copub_col, ville_col]).iterrows():
        G.add_node(row[auteurs_fr_col], type="Inria")
        G.add_node(row[auteurs_copub_col], type="Copubliant")
        G.add_node(row[ville_col], type="Ville")
        # edges between Inria author and copub, copub and ville
        G.add_edge(row[auteurs_fr_col], row[auteurs_copub_col])
        G.add_edge(row[auteurs_copub_col], row[ville_col])
    pos = nx.spring_layout(G, k=0.3, iterations=10, seed=42)
    return G, pos

@st.cache_data
def make_wordcloud(text):
    bg = "#004280" if is_dark else "white"
    wc = WordCloud(width=800, height=400, background_color=bg, colormap="tab10").generate(text)
    return wc

# -------------------
# Titre principal (colored via markdown safe)
# -------------------
st.markdown(f"<h1 style='color:{PRIMARY_COLOR}'>Copublications d'auteurs Inria (Sophia & Bordeaux)</h1>", unsafe_allow_html=True)

# -------------------
# Tabs
# -------------------
tab1, tab2, tab3, tab4 = st.tabs(["Visualisation générale", "Réseau copublication", "Carte du monde", "Contact"])

# -------------------
# Tab 1 : Dashboard
# -------------------
with tab1:
    st.markdown(f"<h2 style='text-align:center;color:{SECONDARY_COLOR};'>KPI et Dashboard</h2>", unsafe_allow_html=True)

    pubs_year = compute_yearly(df_filtered)
    total_pubs = pubs_year[hal_col].sum() if not pubs_year.empty else 0
    total_villes = df_filtered[ville_col].nunique()
    total_auteurs_inria = df_filtered[auteurs_fr_col].nunique()
    total_auteurs_copub = df_filtered[auteurs_copub_col].nunique()

    pubs_par_centre = df_filtered.groupby(centre_col)[hal_col].nunique() if not df_filtered.empty else pd.Series(dtype=int)
    pubs_bordeaux = df_filtered[df_filtered[ville_col] == "Bordeaux"][hal_col].nunique() if "Bordeaux" in df_filtered[ville_col].unique() else 0
    pubs_sophia = df_filtered[df_filtered[ville_col] == "Sophia"][hal_col].nunique() if "Sophia" in df_filtered[ville_col].unique() else 0

    delta_pubs = (pubs_year[hal_col].iloc[-1] - pubs_year[hal_col].iloc[-2]) if len(pubs_year) > 1 else 0

    # KPI style (HTML blocks)
    kpi_style = f"""
    <div style="background:{NEUTRAL_COLOR}; border-radius: 25px; padding: 18px; text-align:center;
                box-shadow: 3px 3px 10px rgba(0,0,0,0.15); font-weight:bold; color:{TEXT_COLOR}; margin:8px;">
        {{title}}<br><span style='font-size:26px;color:{PRIMARY_COLOR}'>{{value}}</span>{{delta}}
    </div>
    """

    kpi_cols = st.columns(7)
    kpi_cols[0].markdown(
        kpi_style.format(title="Publications", value=total_pubs, delta=f"<br><span style='color:{ACCENT_COLOR}'>{'+' if delta_pubs>=0 else ''}{delta_pubs}</span>"),
        unsafe_allow_html=True
    )
    kpi_cols[1].markdown(kpi_style.format(title="Villes", value=total_villes, delta=""), unsafe_allow_html=True)
    kpi_cols[2].markdown(kpi_style.format(title="Auteurs Inria", value=total_auteurs_inria, delta=""), unsafe_allow_html=True)
    kpi_cols[3].markdown(kpi_style.format(title="Auteurs copubliants", value=total_auteurs_copub, delta=""), unsafe_allow_html=True)
    kpi_cols[4].markdown(kpi_style.format(title="Publications par centre", value=pubs_par_centre.sum() if not pubs_par_centre.empty else 0, delta=""), unsafe_allow_html=True)
    kpi_cols[5].markdown(kpi_style.format(title="Bordeaux", value=pubs_bordeaux, delta=""), unsafe_allow_html=True)
    kpi_cols[6].markdown(kpi_style.format(title="Sophia", value=pubs_sophia, delta=""), unsafe_allow_html=True)

    # Publications par année (Plotly)
    st.subheader("Publications par année")
    if not pubs_year.empty:
        fig_year = px.bar(
            pubs_year,
            x=annee_col,
            y=hal_col,
            text=hal_col,
            color_discrete_sequence=[PRIMARY_COLOR]
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
    else:
        st.info("Aucune publication par année à afficher.")

    # TOP 10
    st.subheader("TOP 10")
    top_villes = compute_top(df_filtered, ville_col)
    top_orgs = compute_top(df_filtered, org_col)
    col1, col2 = st.columns(2)

    # Pie TOP villes
    if not top_villes.empty:
        colors_v = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, "#8c8c8c", "#d9d9d9"][:len(top_villes)]
        fig_villes = go.Figure(data=[go.Pie(
            labels=top_villes.index,
            values=top_villes.values,
            hole=0.4,
            marker_colors=colors_v,
            pull=[0.05]*len(top_villes),
            textinfo='label+percent')])
        fig_villes.update_layout(title="Villes copubliantes", title_x=0.5, font=dict(color=TEXT_COLOR),
                                 plot_bgcolor=BACKGROUND_COLOR, paper_bgcolor=BACKGROUND_COLOR)
        col1.plotly_chart(fig_villes, use_container_width=True)
    else:
        col1.info("Aucune ville à afficher.")

    # Pie TOP organismes
    if not top_orgs.empty:
        colors_o = [PRIMARY_COLOR, ACCENT_COLOR, SECONDARY_COLOR, "#8c8c8c", "#d9d9d9"][:len(top_orgs)]
        fig_orgs = go.Figure(data=[go.Pie(
            labels=top_orgs.index,
            values=top_orgs.values,
            hole=0.4,
            marker_colors=colors_o,
            pull=[0.05]*len(top_orgs),
            textinfo='label+percent')])
        fig_orgs.update_layout(title="Organismes copubliants", title_x=0.5, font=dict(color=TEXT_COLOR),
                               plot_bgcolor=BACKGROUND_COLOR, paper_bgcolor=BACKGROUND_COLOR)
        col2.plotly_chart(fig_orgs, use_container_width=True)
    else:
        col2.info("Aucun organisme à afficher.")

    # WordCloud (if present)
    if "Mots-cles" in df_filtered.columns:
        if st.button("Générer le WordCloud"):
            text = " ".join(df_filtered["Mots-cles"].dropna().astype(str))
            if text:
                wc = WordCloud(width=800, height=400,
                               background_color=BACKGROUND_COLOR if not is_dark else "#004280",
                               colormap="tab10").generate(text)
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)

# -------------------
# Tab 2 : Réseau de copublication
# -------------------
with tab2:
    # Use st.markdown for colored header (st.header with unsafe causes TypeError)
    st.markdown(f"<h2 style='color:{PRIMARY_COLOR}'>Réseau de copublication</h2>", unsafe_allow_html=True)

    if st.button("Générer le réseau"):
        max_nodes = 200
        subset = df_filtered.head(max_nodes)

        # Create graph
        G = nx.Graph()
        for _, row in subset.dropna(subset=[auteurs_fr_col, auteurs_copub_col, ville_col]).iterrows():
            # Add nodes (use safe presence checks)
            if pd.notna(row.get(centre_col)):
                G.add_node(row[centre_col], type="Centre")
            if pd.notna(row.get(equipe_col)):
                G.add_node(row[equipe_col], type="Equipe")
            if pd.notna(row.get(auteurs_fr_col)):
                G.add_node(row[auteurs_fr_col], type="Auteur_FR")
            if pd.notna(row.get(auteurs_copub_col)):
                G.add_node(row[auteurs_copub_col], type="Auteur_CP")
            if pd.notna(row.get(ville_col)):
                G.add_node(row[ville_col], type="Ville")

            # Edges (guarded)
            try:
                if pd.notna(row.get(centre_col)) and pd.notna(row.get(equipe_col)):
                    G.add_edge(row[centre_col], row[equipe_col])
                if pd.notna(row.get(equipe_col)) and pd.notna(row.get(auteurs_fr_col)):
                    G.add_edge(row[equipe_col], row[auteurs_fr_col])
                if pd.notna(row.get(auteurs_fr_col)) and pd.notna(row.get(auteurs_copub_col)):
                    G.add_edge(row[auteurs_fr_col], row[auteurs_copub_col])
                if pd.notna(row.get(auteurs_copub_col)) and pd.notna(row.get(ville_col)):
                    G.add_edge(row[auteurs_copub_col], row[ville_col])
            except Exception:
                # skip problematic row
                continue

        if G.number_of_nodes() == 0:
            st.warning("Aucun nœud à afficher dans le réseau.")
        else:
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

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
                mode="lines",
                showlegend=False
            )

            color_map = {
                "Centre": PRIMARY_COLOR,
                "Equipe": ACCENT_COLOR,
                "Auteur_FR": SECONDARY_COLOR,
                "Auteur_CP": ACCENT_COLOR,
                "Ville": "#8c8c8c"
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
                        node_text.append(f"{node} ({node_type}) - {node_degree.get(node, 0)} copubs")
                        node_size.append(8 + node_degree.get(node, 0) * 2)
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

            fig_net = go.Figure(data=[edge_trace] + node_traces,
                                 layout=go.Layout(
                                     title=dict(text="Réseau des copublications", x=0.5, font=dict(color=PRIMARY_COLOR)),
                                     showlegend=True,
                                     legend=dict(title="Type de nœud", font=dict(color=TEXT_COLOR)),
                                     hovermode="closest",
                                     plot_bgcolor=BACKGROUND_COLOR,
                                     paper_bgcolor=BACKGROUND_COLOR,
                                     font=dict(color=TEXT_COLOR)
                                 ))
            st.plotly_chart(fig_net, use_container_width=True)

# -------------------
# Tab 3 : Carte (pydeck)
# -------------------
with tab3:
    st.markdown(f"<h2 style='color:{SECONDARY_COLOR}'>Carte des copublications</h2>", unsafe_allow_html=True)
    if st.button("Générer la carte"):
        if "Longitude" not in df_filtered.columns or "Latitude" not in df_filtered.columns:
            st.warning("Les colonnes 'Latitude' et 'Longitude' sont absentes.")
        else:
            df_map = df_filtered.dropna(subset=["Latitude", "Longitude"])
            if df_map.empty:
                st.warning("Aucune donnée valide pour tracer la carte.")
            else:
                inria_centers = [
                    {"name": "Bordeaux", "lat": 44.833328, "lon": -0.56667, "color": [2,48,89]},
                    {"name": "Sophia", "lat": 43.6200, "lon": 7.0500, "color": [191,17,17]}
                ]
                if centres:
                    inria_centers = [c for c in inria_centers if c["name"].lower() in [cc.lower() for cc in centres]]

                heatmap_df = pd.DataFrame({"lon": df_map["Longitude"], "lat": df_map["Latitude"]})
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
                    map_style=pdk.map_styles.CARTO_DARK if is_dark else pdk.map_styles.CARTO_LIGHT,
                    tooltip={"text": "{name}"}
                )
                st.pydeck_chart(deck)

# -------------------
# Tab 4 : Contact
# -------------------
with tab4:
    st.markdown(f"<h2 style='color:{PRIMARY_COLOR}'>À propos de nous</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='color:{TEXT_COLOR};'>
        Le groupe <b>Datalake</b>, créé en 2022, travaille à rendre possible le croisement de données entre <b>HAL</b> et divers référentiels...
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(f"<h3 style='color:{SECONDARY_COLOR}'>📬 Formulaire de contact</h3>", unsafe_allow_html=True)
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
