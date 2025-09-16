import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
# Fonction envoi mail
# -------------------
def envoyer_mail(nom, email, message):
    sender = "ton_adresse_mail@inria.fr"  # à remplacer par une adresse expéditrice valide
    receiver = "andrea.nebot@inria.fr"
    subject = f"📩 Nouveau message depuis le dashboard (de {nom})"

    body = f"""
    Nom : {nom}
    Email : {email}
    Message :
    {message}
    """

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.inria.fr", 587) as server:  # adapte serveur/port si besoin
            server.sendmail(sender, receiver, msg.as_string())
        return True
    except Exception as e:
        print("Erreur envoi mail:", e)
        return False

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
st.markdown(f"<h1 style='color:{PRIMARY_COLOR}'>Copublications d'auteurs Inria (Sophia & Bordeaux) avec l'Italie</h1>", unsafe_allow_html=True)

# -------------------
# Tabs
# -------------------
tab1, tab2, tab3, tab4 = st.tabs(["Visualisation générale", "Réseau copublication", "Carte Italie", "Contact"])

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

    pubs_year = compute_yearly(df_filtered)
    fig_year = px.bar(pubs_year, x=annee_col, y=hal_col, title="Publications par année",
                      color=hal_col, color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig_year, use_container_width=True)

    # Top villes (couleurs variées)
    top_villes = compute_top(df_filtered, ville_col)
    fig_villes = go.Figure(data=[go.Pie(labels=top_villes.index, values=top_villes.values, hole=0.4)])
    fig_villes.update_traces(marker=dict(colors=px.colors.qualitative.Pastel))
    st.plotly_chart(fig_villes, use_container_width=True)

    # Top organismes (couleurs variées)
    top_orgs = compute_top(df_filtered, org_col)
    fig_orgs = go.Figure(data=[go.Pie(labels=top_orgs.index, values=top_orgs.values, hole=0.4)])
    fig_orgs.update_traces(marker=dict(colors=px.colors.qualitative.Set3))
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
# Onglet 4 : Contact
# -------------------
with tab4:
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
                if envoyer_mail(nom, email, message):
                    st.success(f"Merci {nom} ! Votre message a bien été envoyé ✅")
                else:
                    st.error("❌ Une erreur est survenue lors de l'envoi du message.")
