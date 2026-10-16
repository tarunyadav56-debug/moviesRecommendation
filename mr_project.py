import streamlit as st
import joblib
import requests

model=joblib.load('model.pkl')
vectors=joblib.load('vectors.pkl')
df=joblib.load('dataframe.pkl')

st.set_page_config(layout='wide')

st.markdown("""
<style>
.banner {
    background: linear-gradient(90deg, #4aaaaa, #000000);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 35px;
    font-weight: bold;
    margin-bottom: 25px;
}
</style>

<div class="banner">
    Movie Recommendation Project
</div>
""", unsafe_allow_html=True)

st.sidebar.image("flag.jpg")
st.sidebar.title("About us")
st.sidebar.text("We are developing ML projects based on NLP in LN AI Academy")

st.sidebar.title("About Project")
st.sidebar.text("This project recommmend movie based on story write director of watched movie")

st.sidebar.title("Contact us")
st.sidebar.text("+919717776422")

watched_movie=st.selectbox("Select Watched Movie",options=df['name'])
if st.button("Predict",key="b1"):
    index=df[df.name==watched_movie].index[0]
    distances,indexes=model.kneighbors(vectors[index].toarray())
    for i in indexes[0][1:]:
        st.text(df.loc[i].values[1])
        mid=df.loc[i].values[0]
        resp=requests.get(f"http://www.omdbapi.com/?i=tt3896198&apikey=7472611b")
        data=resp.json()
        st.image(data['Poster'],width=300)   

                
        
