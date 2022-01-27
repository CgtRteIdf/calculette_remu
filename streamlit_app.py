import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import datetime
import plotly

from data import *
from utils import *

pd.options.plotting.backend = "plotly"

st.set_page_config(layout="wide")

st.title('Calculette rémunération')
st.write('Bienvenue ! Cette calculette permet de projeter sa rémunération actuelle (**brute**) dans le temps, de **simuler les évolutions prévues** (prise de NR par ex) et de **comparer cette projection avec l’inflation**. **L\'application n\'enregistre et n\'envoie aucune donnée personnelle.**')
st.write('Pour mémoire l’inflation n’est pas le même indice que le pouvoir d’achat, car elle ne comprend pas les coûts de logement, par exemple. En revanche, elle donne un bon premier aperçu de ce que représenterait votre rémunération d’aujourd’hui avec les prix de demain.')
texte = ['Dans la partie latérale gauche, vous pouvez définir différents paramètres pour vos simulations :\n',
	 '- **NR :** votre NR actuel\n',
	 '- **Taux de promotion :** le rythme auquel vous imaginez avoir une promotion (prise d’un NR). La courbe « optimiste » prend une vision légèrement positive que la moyenne, la courbe pessimiste c’est l’inverse. La calculette ne prend pas en compte les NR « Hors Classe ».\n', 
	 '- **Ancienneté :** intègre les prises d’échelon automatiquement.\n',
	 '- **Taux d’inflation :** à ajuster selon les prévisions INSEE.']
texte = ' '.join(texte)

st.write(texte)
st.write('**L\'application n\'enregistre et n\'envoie aucune donnée personnelle.**')


date_actuelle = datetime.date.today().year

# paramètres
chart_list = ["rémunération brute", "historique SNB", "Obsolète : rémunération avec IC"]
chart_id = "rémunération brute" # st.sidebar.selectbox("type de graphique :", chart_list, index=0)

NR_actuel = int(st.sidebar.number_input("NR actuel", min_value=0, max_value=370, value=150, step=5))
# GF_actuel = st.sidebar.number_input("GF actuel", min_value=0, max_value=19, value=10)
date_entree_ieg = int(st.sidebar.number_input("année d'entrée dans les IEG", min_value=1970, max_value=date_actuelle, value=date_actuelle, step=1))
echelon_actuel = int(st.sidebar.number_input("échelon", min_value=4, max_value=12, value=4, step=1))
# nombre d'années entre deux NR
periode_NR_moyenne = st.sidebar.number_input("nombre d'années entre 2 promotions (moyenne)", min_value=0.1, max_value=20., value=float(periode_moyenne), step=0.5)
# nombre d'années entre deux NR
periode_NR_optimiste = st.sidebar.number_input("nombre d'années entre 2 promotions (optimiste)", min_value=0.1, max_value=20., value=float(np.round(periode_moyenne)), step=0.5)
# nombre d'années entre deux NR
periode_NR_pessimiste = st.sidebar.number_input("nombre d'années entre 2 promotions (pessimiste)", min_value=0.1, max_value=20., value=float(np.round(2*periode_moyenne)), step=0.5)

# annee
anciennete = date_actuelle - date_entree_ieg
taux_inflation = 0.01 * st.sidebar.slider('Taux d\'inflation', 0., 4, inflation_moyenne, 0.1, format="%f%%")
#date_depart_retraite = int(st.sidebar.number_input("année de départ en retraite", min_value=2021, max_value=2065, value=int(date_actuelle + 42 - anciennete), step=1))
date_depart_retraite = int(st.sidebar.number_input("année de départ en retraite", min_value=2021, max_value=2065, value=int(2050), step=1))
# college = st.sidebar.radio(
#     "college",
#     ('Execution', 'Maîtrise', 'Cadre')
# )
# genre = st.sidebar.radio(
#     "genre",
#     ('Femme', 'Homme', 'Ne se prononce pas')
# )

# calculs
# initialisation
data_grille = "data/grille_salariale.csv"
data_snb="data/SNB.csv"
salaire_net_actuel = calc_rem(echelon_actuel, NR_actuel, data=data_grille)

# trajectoire moyenne
df_moyenne, __ = calc_trajectoire(NR_actuel, echelon_actuel, periode_NR_moyenne, taux_inflation, date_actuelle, date_entree_ieg, date_depart_retraite, data_grille=data_grille, data_snb=data_snb)  
df_optimiste, trajectoire = calc_trajectoire(NR_actuel, echelon_actuel, periode_NR_optimiste, taux_inflation, date_actuelle, date_entree_ieg, date_depart_retraite, data_grille=data_grille, data_snb=data_snb)
df_pessimiste, __ = calc_trajectoire(NR_actuel, echelon_actuel, periode_NR_pessimiste, taux_inflation, date_actuelle, date_entree_ieg, date_depart_retraite, data_grille=data_grille, data_snb=data_snb)

# st.write(df_moyenne.head())

if chart_id == chart_list[0]:
    st.subheader('Trajectoires')
    st.write('Les trajectoires moyennes sont obtenues sous l\'hypothèse que chaque agent·e RTE a la même probabilité de recevoir un avancement chaque année. **On obtient donc en moyenne un avancement tous les 2,4 ans.**')
    col1, col2 = st.columns(2)
    
    cols = df_moyenne.columns
    fig = simu_plot(
        dict(zip(["moyenne", "pessimiste", "optimiste"],
            [df_moyenne, df_pessimiste, df_optimiste])), 
        cols[1],
        legende="simulations",
        title="Rémunérations brutes non compensées de l'inflation",
        axe_y="euros"
        )
    col1.plotly_chart(fig,use_container_width = True)
    
    fig = simu_plot(
        dict(zip(["moyenne", "pessimiste", "optimiste"],
            [df_moyenne, df_pessimiste, df_optimiste])), 
        cols[2],
        legende="simulations",
        title="Rémunérations brutes en euros constants",
        axe_y="euros constants"
        )
    col2.plotly_chart(fig,use_container_width = True)

elif chart_id == chart_list[1]:
    st.subheader('Historique SNB')
    df_snb = pd.read_csv(data_snb)
    df_snb = df_snb.iloc[:len(df_snb) - 1].set_index("Année")
    fig = df_snb.plot()
    st.plotly_chart(fig)
elif chart_id == chart_list[2]:
    st.subheader('Trajectoires')
    fig = line_plot_ci(df_moyenne, df_pessimiste, df_optimiste)#0.9*df_moyenne, 1.1*df_moyenne)

    #st.write(fig)
    st.plotly_chart(fig, use_container_width=True)


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(trajectoire)

# st.download_button(
# "Télécharger les données de ma simulation",
# csv,
# "Données_calculette_rémunération.csv",
# "text/csv",
# key='download-csv'
# )

st.write('Cette calculette est rendue accessible par le travail d’adhérent·es bénévoles. Vous pouvez consulter les données et le code source de cette application sur [GitHub](https://github.com) ou nous faire part de vos questions et suggestions à [cgtrterp@gmail.com](mailto:cgtrterp@gmail.com).')


# cacher le "Made with streamlit
# https://www.kdnuggets.com/2021/07/streamlit-tips-tricks-hacks-data-scientists.html
hide_streamlit_style = """
	<style>
	/* This is to hide Streamlit footer */
	footer {visibility: hidden;}
	/*
	</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


hide_count = """
	<style>
	.count {
    visibility: hidden;
    }
	</style>
"""
import streamlit.components.v1 as components
from pathlib import Path



my_file = Path("count_df.csv")

if not my_file.is_file():
    df_count = pd.DataFrame(columns=["count"])
    df_count["count"] = [0]
else:
    df_count = pd.read_csv(my_file)

df_count["count"] = df_count["count"] + 1
df_count.to_csv(my_file)
count = df_count["count"][0]
count = """<div class="count">""" + "compte" + str(count) + """</h4>"""
st.markdown(hide_count, unsafe_allow_html=True)
#components.html(count)
st.markdown(count, unsafe_allow_html=True)





