# Let's write the app code

# 1- Import Essential Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# Import custom CSS for dark theme
st.markdown('<link href="theme/custom.css" rel="stylesheet">', unsafe_allow_html=True)

#
st.title("`Music Genere Classification App🎸`")

# display the video
st.video('Images/vid2.mp4')

# Load music genre classification data
data = pd.read_csv('Data/train.csv', index_col= "Id")
#data

# Sidebar with features
st.sidebar.title('`App Components 🧩💠`')
st.sidebar.header("`1- Select A Feature To View 🧬`")
selected_feature = st.sidebar.selectbox('Features', data.columns)

# Show of attributes
st.header('`Data Overview 👁️`')
st.write(data.sample(10))

# Visualization Data
st.header('`Data Visualization 🚥`')
if st.checkbox('Show Class Visualization 👓'):
    # Class [target]
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    
    # Set the colors    
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
    )
    
    with plt.style.context('dark_background'):

        sns.countplot(data=data, x='Class', ax=ax["A"])
        sns.histplot(data=data, x='Class', color='r', kde=True, ax=ax["B"])
        sns.kdeplot(data=data, x='Class', fill=True, ax=ax["C"])
        sns.scatterplot(data=data, x='Class', y= 'tempo', ax=ax['D'])
        sns.boxplot(data=data, x='Class', showmeans=True, color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    
    st.pyplot(fig)

if st.checkbox('Show energy Visualization 🔎'):
    # energy variable
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ABD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.kdeplot(data =data, x ='energy', hue ='Class', fill =True, palette= 'crest', ax =ax["A"])
        sns.pointplot(data=data, x="Class", y="energy", hue="Class", ax= ax["B"])
        sns.pointplot(data=data, x="Class", y="energy", ax= ax["B"])
        sns.histplot(data=data, x='energy', color ='r', kde=True, ax=ax["C"])
        sns.violinplot(data=data, x="Class", y="energy", ax= ax["D"])
        sns.boxplot(data=data, x= 'energy', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    
    st.pyplot(fig)

st.sidebar.header("`2- Select A ◻️ To Show Visualize`")
if st.sidebar.checkbox('Show duration_in min/ms'):
    # duration_in min/ms 👉🏽 Time of the song
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABB
        ACD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.pointplot(data=data, x="Class", y="duration_in min/ms", ax= ax["A"])
        sns.pointplot(data=data, x="Class", y="duration_in min/ms", hue="Class", ax= ax["A"])
        sns.kdeplot(data =data, x ='duration_in min/ms', fill =True, color= 'r', ax =ax["B"])
        sns.histplot(data=data, x=(data['duration_in min/ms']/1e6), kde=True, ax=ax["C"])
        sns.violinplot(data=data, x= "duration_in min/ms", ax= ax["D"])
        sns.boxplot(data=data, x= 'duration_in min/ms', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

if st.sidebar.checkbox('Show Loudness'):
    # loudness
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.violinplot(data=data, x= "loudness", ax= ax["A"])
        sns.histplot(data=data, x='loudness', color= 'r', kde=True, ax=ax["B"])
        sns.kdeplot(data=data, x='loudness', color= 'g', fill=True, ax=ax["C"])
        sns.pointplot(data=data, x="Class", y="loudness", ax= ax["D"])
        sns.pointplot(data=data, x="Class", y="loudness", hue="Class", ax= ax["D"])
        sns.boxplot(data=data, x= 'loudness', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

if st.sidebar.checkbox('Show Popularity'):
    # Popularity 👉🏻 How much the song is famous?
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.regplot(data= data, x= 'Class', y= 'Popularity', ax=ax["A"])
        sns.histplot(data=data, x='Popularity', color= 'r', kde=True, ax=ax["B"])
        sns.violinplot(data=data, x= "Popularity", ax= ax["C"])
        sns.pointplot(data=data, x="Class", y="Popularity", ax= ax["D"])
        sns.pointplot(data=data, x="Class", y="Popularity", hue="Class", ax= ax["D"])
        sns.boxplot(data=data, x= 'Popularity', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})

    st.pyplot(fig)

if st.sidebar.checkbox('Show Liveness'):
    # liveness
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.pointplot(data=data, x="Class", y="liveness", ax= ax["A"])
        sns.pointplot(data=data, x="Class", y="liveness", hue="Class", ax= ax["A"])
        sns.kdeplot(data =data, x ='liveness', fill =True, color= 'r', ax =ax["B"])
        sns.violinplot(data=data, x= 'liveness', ax=ax["C"])
        sns.regplot(data= data, x= 'Class', y= 'liveness', ax=ax["D"])
        sns.boxplot(data=data, x= 'liveness', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

if st.sidebar.checkbox('Show time_signature'):
    # time_signature
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.pointplot(data=data, x="Class", y="time_signature", ax= ax["A"])
        sns.pointplot(data=data, x="Class", y="time_signature", hue="Class", ax= ax["A"])
        sns.kdeplot(data =data, x ='time_signature', fill =True, color= 'r', ax =ax["B"])
        sns.violinplot(data=data, x= 'time_signature', ax=ax["C"])
        sns.regplot(data= data, x= 'Class', y= 'time_signature', ax=ax["D"])
        sns.boxplot(data=data, x= 'time_signature', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

if st.sidebar.checkbox('Show tempo'):
    # tempo
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.histplot(data=data, x="tempo", kde=True, color='r', ax= ax["A"])
        sns.kdeplot(data =data, x ='tempo', fill =True, color= 'r', ax =ax["B"])
        sns.violinplot(data=data, x= 'tempo', ax=ax["C"])
        sns.regplot(data= data, x= 'Class', y= 'tempo', ax=ax["D"])
        sns.boxplot(data=data, x= 'tempo', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

if st.sidebar.checkbox('Show key'):
    # key
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.histplot(data=data, x="key", kde=True, color='r', ax= ax["A"])
        sns.kdeplot(data =data, x ='key', fill =True, color= 'g', ax =ax["B"])
        sns.violinplot(data=data, x= 'key', ax=ax["C"])
        sns.regplot(data= data, x= 'Class', y= 'key', ax=ax["D"])
        sns.boxplot(data=data, x= 'key', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

if st.sidebar.checkbox('Show instrumentalness'):
    # instrumentalness
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.regplot(data= data, x= 'Class', y= 'instrumentalness', ax=ax["A"])
        sns.kdeplot(data =data, x ='instrumentalness', fill =True, color= 'r', ax =ax["B"])
        sns.violinplot(data=data, x= 'instrumentalness', ax=ax["C"])
        sns.pointplot(data=data, x="Class", y="instrumentalness", ax= ax["D"])
        sns.pointplot(data=data, x="Class", y="instrumentalness", hue="Class", ax= ax["D"])
        sns.boxplot(data=data, x= 'instrumentalness', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})

    st.pyplot(fig)
if st.sidebar.checkbox('Show mode'):
    # mode
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.countplot(data= data, x= 'mode', ax=ax["A"])
        sns.kdeplot(data =data, x ='mode', fill =True, color= 'r', ax =ax["B"])
        sns.violinplot(data=data, x= 'mode', ax=ax["C"])
        sns.pointplot(data=data, x="Class", y="mode", ax= ax["D"])
        sns.pointplot(data=data, x="Class", y="mode", hue="Class", ax= ax["D"])
        sns.boxplot(data=data, x= 'mode', showmeans= True,
                color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

if st.sidebar.checkbox('Show speechiness'):
    # speechiness
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.countplot(data= data, x= 'speechiness', ax=ax["A"])
        sns.kdeplot(data =data, x ='speechiness', fill =True, color= 'r', ax =ax["B"])
        sns.violinplot(data=data, x= 'speechiness', ax=ax["C"])
        sns.pointplot(data=data, x="Class", y="speechiness", ax= ax["D"])
        sns.pointplot(data=data, x="Class", y="speechiness", hue="Class", ax= ax["D"])
        sns.boxplot(data=data, x= 'speechiness', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})

    st.pyplot(fig)

if st.sidebar.checkbox('Show valence'):
    # valence 👉🏼 A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    
    with plt.style.context('dark_background'):
        sns.regplot(data= data, x= 'Class', y= 'valence', color= 'g', ax=ax["A"])
        sns.kdeplot(data =data, x ='valence', fill =True, color= 'r', ax =ax["B"])
        sns.violinplot(data=data, x= 'valence', ax=ax["C"])
        sns.pointplot(data=data, x="Class", y="valence", ax= ax["D"])
        sns.pointplot(data=data, x="Class", y="valence", hue="Class", ax= ax["D"])
        sns.boxplot(data=data, x= 'valence', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

if st.sidebar.checkbox('Show danceability'):
    # danceability
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.pointplot(data=data, x="Class", y="danceability", ax= ax["A"])
        sns.pointplot(data=data, x="Class", y="danceability", hue="Class", ax= ax["A"])
        sns.histplot(data =data, x ='danceability', kde= True, color= 'r', ax =ax["B"])
        sns.violinplot(data=data, x= 'danceability', ax=ax["C"])
        sns.scatterplot(data= data, x= 'danceability', y= 'valence', ax=ax["D"])
        sns.lineplot(data=data, x='danceability', y='valence', color= 'r', ax= ax["D"])
        sns.boxplot(data=data, x= 'danceability', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

if st.sidebar.checkbox('Show acousticness'):
    # acousticness
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('0.1')
    ax = fig.subplot_mosaic(
        """
        ABC
        ADD
        EEE
        """
        )
    with plt.style.context('dark_background'):
        sns.pointplot(data=data, x="Class", y="acousticness", ax= ax["A"])
        sns.pointplot(data=data, x="Class", y="acousticness", hue="Class", ax= ax["A"])
        sns.histplot(data =data, x ='acousticness', kde= True, color= 'r', ax =ax["B"])
        sns.violinplot(data=data, x= 'acousticness', ax=ax["C"])
        sns.scatterplot(data= data, x= 'acousticness', y= 'instrumentalness', ax=ax["D"])
        sns.boxplot(data=data, x= 'acousticness', showmeans= True,
                    color='deeppink', ax=ax["E"], notch=True, medianprops={"color": "w"})
    st.pyplot(fig)

# Multiclassification task
st.header('`Music Genre Classification ✂️`')
def perform_feature_engineering(data):
    # Fill missing values in 'Popularity' column with mean
    data['Popularity'] = data['Popularity'].fillna(data['Popularity'].mean())

    # Fill missing values in 'key' column with mode
    data['key'] = data['key'].fillna(data['key'].mode()[0])

    # Fill missing values in 'instrumentalness' column with mean
    data['instrumentalness'] = data['instrumentalness'].fillna(data['instrumentalness'].mean())

    # Drop categorical columns 'Artist Name' and 'Track Name'
    data = data.drop(['Artist Name', 'Track Name'], axis=1)

    # Set the target
    y = data['Class']
    X = data.drop(['Class'], axis=1)
    
    return X, y

X, y = perform_feature_engineering(data)

# test data
def preprocess_data(test):
    # Read the CSV file into a DataFrame
    test = pd.read_csv(test, index_col="Id")
    
    # Drop 'Artist Name' and 'Track Name' columns
    test = test.drop(['Artist Name', 'Track Name'], axis=1)
    
    # Fill missing values
    test['Popularity'] = test['Popularity'].fillna(test['Popularity'].mean())
    test['key'] = test['key'].fillna(test['key'].mode()[0])
    test['instrumentalness'] = test['instrumentalness'].fillna(test['instrumentalness'].mean())
    
    return test
test = preprocess_data('Data/test.csv')

#
st.sidebar.header("`3- Determine The Class Based On👇🏼`")
# Create sliders for music genre features
col1, col2 = st.sidebar.columns(2)
with col1:
    popularity = st.slider('Popularity', 0, 100, 50)
    danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
    energy = st.slider('Energy', 0.0, 1.0, 0.5)
    key = st.slider('Key', 1, 11, 1)  # Key should be an integer, not a floating-point value
    loudness = st.slider('Loudness', -39, 1, 1)  # Changed max_value to 1 to match step=0.5
    mode = st.slider('Mode', 0, 1, 1)  # Mode should be an integer, not a floating-point value
    speechiness = st.slider('Speechiness', 0.0, 1.0, 0.5)

with col2:
    acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
    instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.5)
    liveness = st.slider('Liveness', 0.0, 1.0, 0.5)
    valence = st.slider('Valence', 0.0, 1.0, 0.5)
    tempo = st.slider('Tempo', 30, 220, 50)
    duration = st.slider('Duration', 1, 20000, 100)
    time_signature = st.slider('Time Signature', 1, 5, 1)

# multiclassification model code
model = RandomForestClassifier()
model.fit(X, y)

classification_result = model.predict(test)
st.write(classification_result)


st.sidebar.header("`4- Predict Classes Section 🧎`")
# Simulate a prediction button
if st.sidebar.button("Predict Music Genre Class 🛎️"):
    # Format the input features as a 2D array
    test_data = np.array([[popularity, danceability, energy, key, loudness, mode,
                           speechiness, acousticness, instrumentalness, liveness,
                           valence, tempo, duration, time_signature]])
    # Make the prediction
    prediction = model.predict(test_data)    
    predicted_genre = str(prediction[0])

    # Show the result
    st.header("`Prediction Result 🧵`")
    st.success("Predicted Music Genre 🧱: " + predicted_genre)
    st.sidebar.header("`Prediction Result 🧵`")
    st.sidebar.success("Predicted Music Genre 🧱: " + predicted_genre)