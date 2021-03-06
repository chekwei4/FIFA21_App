#first commit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import logging
from sys import exit
from PIL import Image
import base64
from bokeh.models.widgets import Div

st.set_page_config(layout="wide", page_title="Chek's FIFA21 App")
image = Image.open('cover_pic.jpeg')
st.image(image)
st.title("FIFA21 Players Analysis⚽")
st.markdown("This app explores unsupervised clustering techniques to cluster **FIFA21 players**!")

"""
1. Choose between clustering techniques
2. Select Number of Clusters
3. Set players' overall range
4. Observe which players are "more similar" than the rest
---
"""

col1, col2 = st.columns(2)
open_colab = col1.button("🚀 EDA in Google Colab") 
open_github = col1.button("💻 Project Github") 

if open_colab:
    print("opening eda...")
    # webbrowser.open_new_tab("https://colab.research.google.com/drive/1TSczMMp1Rya-ib6tE__lsP32pqm9TB45?usp=sharing")
    js = "window.open('https://colab.research.google.com/drive/1gtVNCDrjNk1yQCwCL21zIPHLkp7LdWvV?usp=sharing')"
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

if open_github:
    print("opening git...")
    # webbrowser.open_new_tab("https://github.com/chekwei4/Titanic_App")
    js = "window.open('https://github.com/chekwei4/FIFA21_App')"
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

col1 = st.sidebar
col1.header('Clustering Techniques')
cluster_method = col1.selectbox('Select Clustering Techniques', ('KMeans', 'Hierarchical'))

col1.header('Select Teams')
seed = col1.radio("Select Number of Clusters", ("3 - GK, Defender, Forward", "4 - GK, Defender, Midfielder, Forward", "5 - GK, Defender, D.Midfielder, A.Midfielder, Forward"))

col1.header('Players Overall')
overall_range = col1.slider(label="Overall Range:", min_value=0, value=(75, 85), max_value=99)
low_range = overall_range[0]
up_range = overall_range[1]

col1.markdown('___')
col1.text("Thanks for dropping by...")
col1.markdown('Created by [Chekwei](https://github.com/chekwei4/)')
col1.markdown('Other projects [Here](https://chekwei4.github.io/Chek_Wei_Portfolio/)')
print("code runs successfully...")

def main():
    try:
        df = pd.read_csv("./data/data.csv")
    except IOError:
        logging.exception('data not found...preparing data now...')
        prepare_data()
    try:
        df = filter_players_overall(df, low_range, up_range)
        df_clustered = run_clustering(df, cluster_method)
        # prepare_data()
        features_bring_front = df[["short_name", "overall", "pace", "shooting", "passing", 
                                    "dribbling", "defending", "physic"]]
        df.drop(["short_name", "overall", "pace", "shooting", "passing", 
                                    "dribbling", "defending", "physic"], axis=1, inplace=True)
        # df.insert([0,1], "features_bring_front", features_bring_front)
        df = pd.concat([features_bring_front, df], axis=1)
        st.subheader("Players Data")
        st.dataframe(df)
        download = st.button(label="💾 Download CSV")
        if download:
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings
            href = f'<a href="data:file/csv;base64,{b64}" download="players.csv">Download csv file</a>'
            st.markdown(file_download(df), unsafe_allow_html=True)
        if df_clustered is not None:
            fig = plot_cluster(df_clustered)
            st.subheader("Display Clustering")
            st.plotly_chart(fig)
        else:
            st.text("Clustering technique is not ready...come back soon!")
    except ValueError:
        logging.exception('pls select wider range for overall')
        st.text("pls select wider range for overall...")

def file_download(df):
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="players.csv" target="_blank">Export CSV here</a>'
    return href

def plot_cluster(df_clustered):
    fig = px.scatter(df_clustered, x="X", y="y", color="label", hover_name="short_name")
    fig.update_traces(marker=dict(size=14,line=dict(width=1,color='DarkSlateGrey')),selector=dict(mode='markers'))
    fig.update_layout(title='FIFA21 Players')
    fig.update_layout(
        autosize=False,
        width=800,
        height=800)
    # fig.show()
    return fig

def run_clustering(df, cluster_method):
    # df = prepare_data()
    # df = pd.read_csv("./data/data.csv")
    df_scaled = scale_data(df)
    df_pca = apply_pca(df_scaled)
    df_pca = pd.DataFrame(df_pca)
    df_clustered = None
    if cluster_method == "KMeans":
        seed_num = 0
        if seed == "3 - GK, Defender, Forward":
            seed_num=3
        if seed == "4 - GK, Defender, Midfielder, Forward":
            seed_num=4
        if seed == "5 - GK, Defender, D.Midfielder, A.Midfielder, Forward":
            seed_num=5
        kmeans = KMeans(n_clusters=seed_num).fit(df_pca)
        labels = kmeans.labels_
        labels = pd.DataFrame(labels)
        df_clustered = pd.concat([df_pca, labels, df["short_name"]], axis=1)
        df_clustered.columns = ["X", "y", "label", "short_name"]
    elif cluster_method == "Hierarchical":
        logging.exception('clustering technique not ready yet...')
        df_clustered = None
    return df_clustered

def scale_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.drop("short_name", axis=1))
    return df_scaled

def apply_pca(df):
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    return df_pca

def read_data():
    url = "https://drive.google.com/file/d/1AD7U9ExHWk0R3Eg6IQltxy3GDDRzxaEk/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    data = pd.read_csv(path)
    return data

def clean_positional_attributes(df):
    position_attr = ['ls','st','rs', 'lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb', 'rb']
    for position in position_attr:
        df[position] = df[position].apply(lambda x : x.split("+"))
    for position in position_attr:
        df[position] = sum_position_attr(df[position])
        df[position] = df[position].astype('int64')
    return df

def sum_position_attr(col):
    for i in range(len(col)):
        if len(col[i][1]) != 0:
            col[i] = int(col[i][0]) + int(col[i][1])
        else:
            col[i] = int(col[i][0])
    return col

def drop_non_numerical_features(df):
    numerics = ['int16','int32','int64','float16','float32','float64']
    numerical_features = list(df.select_dtypes(include=numerics).columns)
    numerical_features.insert(0,"short_name")
    df = df[numerical_features]
    return df

def drop_features(df):
    df = df.drop(["sofifa_id", "international_reputation", "league_rank", "value_eur", "wage_eur", "release_clause_eur",
                "team_jersey_number", "contract_valid_until", "nation_jersey_number", "defending_marking"], axis=1)
    return df

def data_cleaning(df):
    feature_names = df.columns.tolist()
    print(feature_names)
    del feature_names[feature_names.index("gk_diving"):feature_names.index("gk_diving")+6]
    del feature_names[feature_names.index("pace"):feature_names.index("pace")+6]
    feature_names = feature_names[1:]
    gk_attr = ["gk_diving","gk_handling","gk_kicking" 
            ,"gk_reflexes","gk_speed","gk_positioning"]
    basic_attr = ["pace","shooting", "passing", "dribbling", "defending", "physic"]
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    iterative_imputer = IterativeImputer()
    #df.columns.tolist()[1:]
    col_trans = ColumnTransformer([("simple_imputer", simple_imputer, gk_attr),
                                    ("iterative_imputer", iterative_imputer, basic_attr)],
                                    remainder="passthrough")
    df = col_trans.fit_transform(df)
    col_names = gk_attr + basic_attr + ["short_name"] + feature_names
    return pd.DataFrame(df, columns=col_names)

def filter_players_overall(df,low_range,up_range):
    df = df[(df["overall"]>=low_range) & (df["overall"]<=up_range)]
    return df

def prepare_data():
    df = read_data()
    df = (df.
        pipe(clean_positional_attributes).
        pipe(drop_non_numerical_features).
        pipe(drop_features).
        pipe(data_cleaning))
    df.to_csv("./data/data.csv", index=False)
    # return df

if __name__ == "__main__":
    main()