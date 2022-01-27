import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import plotly.express as px

import datetime
import itertools
from csv import reader

from data import periode_moyenne, taux_NR


def calc_rem(echelon, NR, data="data/grille_salariale.csv"):
    """Calcule le salaire **brut?** pour un échelon et un NR donné, à partir d'une grille de salaires.

    Args:
        echelon (int): échelon pour lequel on calcule la rémunération.
        NR (int): NR pour lequel on calcule la rémunération.
        data (str): string représentant le chemin d'accès vers le fichier csv contenant la grille de salaires.

    Returns:
        float: valeur de la rémunération **brute**
    """
    header = get_header(data, n_rows=2)
    df = pd.read_csv(data, header=2).set_index("NR")
    
    echelons = dict(zip(header[1][1:], df.columns[1:]))
    
    return df[echelons[str(echelon)]][str(NR)]

def calc_trajectoire(NR_actuel, echelon_actuel, periode_NR, taux_inflation, date_actuelle, date_entree_ieg, date_depart_retraite, data_grille="data/grille_salariale.csv", data_snb="data/SNB.csv"):
    # paramètres
    periode = pd.to_timedelta(periode_NR*365/7, unit="W")
    taux_inflation = taux_inflation

    # lecture données
    header = get_header(data_grille, n_rows=2)
    df = pd.read_csv(data_grille, header=2).set_index("NR")
    
    echelons = dict(zip(header[1][1:], df.columns[1:]))

    echelons_anciennete = pd.read_csv("data/relation_echelon_anciennete.csv",header=None).set_index(0).transpose()#.iloc[3:]

    remu_actuelle = calc_rem(echelon_actuel, NR_actuel, data_grille)

    # dates
    date_actuelle = datetime.date.today()
    annee_actuelle = date_actuelle.year

    date_entree_ieg = pd.to_datetime("01/01/"+str(date_entree_ieg))

    # le calcul est approximatif tant qu'on ne demande que l'année de départ en retraite
    date_depart_retraite = date_actuelle + pd.to_timedelta((date_depart_retraite - annee_actuelle)*365/7, unit="W") # a supprimer
    temps_depart_retraite = date_depart_retraite - date_actuelle
    annee_depart_retraite = date_depart_retraite.year

    #dates de changement de NR
    dates_NR = pd.Series(np.arange(0, int(min(
                                      len(df.iloc[np.array(range(len(df))) >= df.index.get_loc(str(NR_actuel))]),
                                      temps_depart_retraite / pd.to_timedelta(periode_NR*365/7, unit="W")
                                    )
                                )
                        ) * periode + pd.to_datetime(date_actuelle))
    dates_NR.index = df.iloc[np.array(range(len(df))) >= df.index.get_loc(str(NR_actuel))][:len(dates_NR)].index

    dates_echelons = pd.to_timedelta(echelons_anciennete["Nombre d'années AVANT l'échelon"]*365/7, unit="W") + date_entree_ieg

    # il faut tenir compte des augmentations d'échelons # obso?
    liste_remu = df[echelons[str(echelon_actuel)]].iloc[np.array(range(len(df))) >= df.index.get_loc(str(NR_actuel))]
        
    df_remu = pd.DataFrame()
    df_remu["dates_NR"] = dates_NR.values
    df_remu["NR"] =  df[echelons[str(echelon_actuel)]].iloc[np.array(range(len(df))) >= df.index.get_loc(str(NR_actuel))].index[:len(dates_NR)]

    # colonnes : NR, coeff, remu, echelon
    # merger avec dates_echelons et dates_NR

    df_remu = df_remu.merge(df, left_on="NR", right_index=True).rename({v: k for k, v in echelons.items()}, axis=1)

    #nouveau calcul de trajectoire
    # présenter un output avec une année par ligne, mais commencer par calculer par dates
    # TODO renommer NR_actuel et echelon_actuel en les mêmes sans les suffices
    trajectoire = pd.DataFrame()

    trajectoire["annees"] = np.arange(annee_actuelle,
                    annee_depart_retraite + 1)

    dates_echelons = pd.Series(dates_echelons)

    trajectoire_remu = pd.DataFrame()
    #trajectoire_remu.columns = ['NR', 'echelon', 'remu']


    NR = NR_actuel, 
    echelon = echelon_actuel
    remu = remu_actuelle

    new_row = pd.Series(data={"NR": str(NR_actuel), "echelon": echelon_actuel, "remu": remu_actuelle}, 
                                name=date_actuelle)
    #append row to the dataframe
    trajectoire_remu = trajectoire_remu.append(new_row, ignore_index=False)

    # https://stackoverflow.com/questions/40146472/quickest-way-to-swap-index-with-values
    dates = pd.concat([pd.Series(dates_echelons.index.values, index=dates_echelons ),
            pd.Series(dates_NR.index.values, index=dates_NR )], axis=1).sort_index()
    dates = dates.rename(dict(zip(dates.columns, ["echelon", "NR"])), axis=1)

    for row in dates.itertuples():
        if ((pd.isnull(row.echelon)) & (not pd.isnull(row.NR))):
            NR_actuel = row.NR
            
            new_row = pd.Series(data={"NR": NR_actuel, 
                                        "echelon": echelon_actuel, 
                                        "remu": df_remu[str(int(echelon_actuel))][df_remu.NR == str(NR_actuel)].values[0]
                                    }, 
                                name=row.Index)
            #append row to the dataframe
            trajectoire_remu = trajectoire_remu.append(new_row, ignore_index=False)

        elif ((not pd.isnull(row.echelon)) & (pd.isnull(row.NR))):
            echelon_actuel = row.echelon
            
            new_row = pd.Series(data={"NR": NR_actuel, 
                                        "echelon": echelon_actuel, 
                                        "remu": df_remu[str(int(echelon_actuel))][df_remu.NR == str(NR_actuel)].values[0]
                                    }, 
                                name=row.Index)
            #append row to the dataframe
            trajectoire_remu = trajectoire_remu.append(new_row, ignore_index=False)
        else:
            print("ERROR", row.echelon, row.NR)
                

    trajectoire_remu.index = pd.to_datetime(trajectoire_remu.index)
    trajectoire_remu = trajectoire_remu.sort_index()

    trajectoire_echelons = pd.DataFrame()
    trajectoire_echelons["date_echelon"] = dates_echelons
    trajectoire_echelons["annee_echelon"] = trajectoire_echelons.date_echelon.dt.year
    trajectoire_echelons["echelon"] = dates_echelons.index

    trajectoire = trajectoire.merge(trajectoire_echelons, left_on="annees", right_on="annee_echelon", how="outer") 

    trajectoire_NR = pd.DataFrame()
    trajectoire_NR = df_remu[['dates_NR', 'NR']]
    trajectoire_NR['annee_NR'] = trajectoire_NR["dates_NR"].dt.year

    trajectoire = (trajectoire.merge(trajectoire_NR, left_on="annees", right_on="annee_NR", how="outer")
                #.dropna(subset=["annees"])
                #.set_index('annees')
                .ffill()
                )
    trajectoire["NR_echelon"] = trajectoire.NR.astype(str) + " " + trajectoire.echelon.astype(str)
    trajectoire_remu["NR_echelon"] = trajectoire_remu.NR.astype(str) + " " +  trajectoire_remu.echelon.astype(str)

    trajectoire = (trajectoire.merge(trajectoire_remu.drop(columns=["NR", "echelon"]), left_on="NR_echelon", right_on="NR_echelon", how="outer")
                .drop(columns=["annee_echelon", "annee_NR", "NR_echelon"])
                .dropna()
                )

    df_snb = pd.read_csv(data_snb)

    evol_moyenne_snb = df_snb["Évolution du SNB"].iloc[:len(df_snb) - 1].mean()
    evol_moyenne_snb

    trajectoire["remu"] = trajectoire.apply(lambda row: row["remu"] * (1 + evol_moyenne_snb /100)**(row["annees"] - annee_actuelle), axis=1)

    trajectoire["inflation"] = trajectoire.apply(lambda row: (1 + taux_inflation)**(row["annees"] - annee_actuelle), axis=1)
    trajectoire["remu_eur_cst"] = trajectoire["remu"] / trajectoire["inflation"]

    # on ne tient pas compte des hors classe
    trajectoire = trajectoire[trajectoire.NR.apply(lambda x: x.isnumeric())]

    df_plot = (trajectoire[["annees", "remu", "remu_eur_cst"]].rename(dict(zip(["annees", "remu", "remu_eur_cst"], ["année", "rémunération", "rémunération en euros constants"])), axis=1).set_index("année", drop=False))
              
    return df_plot, trajectoire

def simu_plot(df_dict, column, legende, title, axe_y):
    
    min_y = 1500 # min([df.to_numpy().min() for df in df_dict.values()])
    max_y = max([df.to_numpy().max() for df in df_dict.values()])
    
    df = pd.DataFrame()
    for k, v in df_dict.items():
        df[k] = v.drop_duplicates(subset=["année"])[column]
        # if len(df) == 0:
        #     df[k] = v[[column, "année"]].rename({column: k}, axis=1).drop_duplicates()
        # else:
        #     df.merge(right=v[column].rename({column: k}, axis=1), left_on="année", right_on="année", how="inner")

    fig = px.line(df, labels={
                     "variable": legende,
                     "value": axe_y
                 },
                 title=title)
    # https://stackoverflow.com/questions/60715706/how-to-remove-trace0-here
    fig.update_traces(hovertemplate='Rémunération : %{y:.0f}€<br>Année : %{x}<extra></extra>')

    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    return fig.update_yaxes(range=[min_y,max_y])

def line_plot_ci(df, df_low=None, df_high=None):

    x = df.index.to_list()
    x_rev = x[::-1]

    fig = go.Figure()

    for col in df.columns:
        y = df[col].to_list()
        y_upper = df_high[col].to_list()
        y_lower = df_low[col].to_list()
        y_lower = y_lower[::-1]

        fig.add_trace(go.Scatter(
            x=x+x_rev,
            y=y_upper+y_lower,
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name=col,# + "(moyenne)",
        ))
    
        fig.add_trace(go.Scatter(
            x=x, y=y,
            #line_color='rgb(0,100,80)',
            name=col,# + "(moyenne)",
        ))

    fig.update_traces(mode='lines')

    fig.update_layout(
    #title="Evolution de la rémunération",
    xaxis_title="Années",
    yaxis_title="Euros",
    legend_title="Rémunération mensuelle nette",
    )

    return fig

def get_header(csv_file, n_rows=1):
    """Returns the first n lines of a csv file as a sequence of strings.

    Args:
        csv_file (string): describes the path to the csv file.
        n_rows (int, optional): defines the number of rows comprised in the header. Defaults to 1.

    Returns:
        sequence of strings: sequence of rows in the header.
    """
    
    content = []
    # open file in read mode
    with open(csv_file, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        # https://www.py4u.net/discuss/247555
        for row in itertools.islice(csv_reader, n_rows):
            # row variable is a list that represents a row in csv
            content.append(row)
    return content

