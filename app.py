#first commit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("FIFA21 Players Analysis")
st.markdown("This app explores unsupervised clustering techniques to cluster **FIFA21 players**!")

col1 = st.sidebar
col1.header('Clustering Techniques')
cluster_method = col1.selectbox('Select Clusterng', ('KMeans', 'Hierarchical'))

col1.header('Select Number of Clusters')
selected_seed = col1.radio("Select Clusters", (3, 4, 5))

col1.header('Players Overall')
overall_range = col1.slider(label="Overall Range:", min_value=0, value=(75, 85), max_value=99)
low_range = overall_range[0]
up_range = overall_range[1]

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/FIFA/players_21.csv")

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

def filter_players_overall(df):
  df = df[df["overall"]>=85]
  return df

df = (df.
      pipe(clean_positional_attributes).
      pipe(drop_non_numerical_features).
      pipe(drop_features).
      pipe(data_cleaning).
      pipe(filter_players_overall))

df.head()