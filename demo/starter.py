import streamlit as st
import re
import ast
import requests
import plotly.express as px
import pandas as pd
from preprocessing import convert2segment, embedding_300, inference
import json
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Demo Data Mining",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# Custom font
with open("/home/nhatdm2k4/Pictures/Data_mining/demo/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html= True)

### Dataframe section ###
### st.cache_data
@st.cache_data
def normalize_text(text):
    text = re.sub('[^A-Za-z]+', ' ', text).lower()
    return re.sub(r'\d', ' ', text)
@st.cache_data
def length_text(text):
    return len(text)
@st.cache_data
def clean_text(text):
  words = text.split()
  words = [x for x in words if x[0] != '#' and x[:4] != 'http']
  return ' '.join(words)
# TODO: implement length sentence distribution
# TODO: implement box plot length distribution
# TODO: implement word cloud
# TODO: implement box plot with most appear words

@st.cache_data
def load_df():
    df = pd.read_csv("/home/nhatdm2k4/Pictures/Data_mining/demo/data/raw_data.csv")
    return df
df = load_df()
df.drop(df[df['content'].isna()].index, inplace=True)
df['content_normalize'] = df['content'].apply(normalize_text)
df['length'] = df['content_normalize'].apply(length_text)
df['label'].replace(2, 0, inplace=True)
@st.cache_data
def load_fixed_df():
    fixed_df = pd.read_csv("/home/nhatdm2k4/Pictures/Data_mining/demo/data/fixed_data.csv")
    return fixed_df
fixed_df = load_fixed_df()

def text2array(text):
  return ast.literal_eval(text)
fixed_df['text_split'] = fixed_df['text_split'].apply(text2array)

st.sidebar.image("/home/nhatdm2k4/Pictures/Data_mining/demo/resource/logo3.png")
st.sidebar.divider()
st.sidebar.markdown('<h1 class="inknut-text">Layout</p>', unsafe_allow_html=True)

with st.sidebar.expander("Custom Layout", expanded=True):
    title_divider = st.checkbox('Title and divider')
    footer_state = st.checkbox('Footer')
if title_divider:
    st.image("/home/nhatdm2k4/Pictures/Data_mining/demo/resource/logo.jpg")
    st.divider()


st.sidebar.markdown('<h1 class="inknut-text">Data</p>', unsafe_allow_html=True)
# Sidebar to show
with st.sidebar.expander("Data Statistics", expanded=True):
    show_data = st.checkbox('Show data')
    distribution = st.checkbox('Labels distribution')
    length_sentence = st.checkbox('Length sentence distribution')
    box_plot_length_sentence = st.checkbox('Box plot length distribution')

with st.sidebar.expander("Plot Data", expanded=True):
    word_cloud = st.checkbox("Word cloud")
    box_plot_most_word = st.checkbox("Box plot most appear words")

st.sidebar.markdown('<h1 class="inknut-text">Demo</h1>', unsafe_allow_html=True)
with st.sidebar.expander("Test Input", expanded=True):
    test_input = st.checkbox('Input text')
    performance_metrics = st.checkbox('Performance metrics')

with st.sidebar.expander("Feature Distribution Visualization", expanded=True):
    tsne_visualization = st.checkbox('T-SNE Visualization')
    pca_visualization = st.checkbox('PCA Visualization')


### Data Section ###
if show_data:
    st.markdown('<h3 class="inknut-text margin-heading">Data Frame</h3>', unsafe_allow_html=True)
    st.write(df)

if distribution:
    st.markdown('<h3 class="inknut-text margin-heading">Label Distribution</h3>', unsafe_allow_html=True)
    label_counts = df['label'].value_counts().to_dict()
    # Pie Chart
    colors = ['#ff9999', '#cc0000']
    fig = px.pie(values=list(label_counts.values()), names=['Bình thường', 'Phản động'], title='Pie chart', color_discrete_sequence=colors)
    fig.update_layout(margin=dict(t=50, b=50, l=50, r=200))
    fig_bar = px.bar(x=['Label 0', 'Label 1'], y=list(label_counts.values()),
                     labels={'x': 'Label', 'y': 'Count'}, title='Bar chart',
                     color=['Bình thường', 'Phản động'], color_discrete_sequence=colors)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(fig_bar)

if length_sentence:
    st.markdown('<h3 class="inknut-text margin-heading">Length sentence distribution</h3>', unsafe_allow_html=True)
    fig = px.histogram(fixed_df, x="length", nbins=40, color_discrete_sequence=['red'],
                       title="Length of Sentences Distribution")
    st.plotly_chart(fig)

if box_plot_length_sentence:
    st.markdown('<h3 class="inknut-text margin-heading">Box plot length distribution</h3>', unsafe_allow_html=True)
    key = list(fixed_df['length'].value_counts().sort_index(ascending=True).keys())
    value = fixed_df['length'].value_counts().sort_index(ascending=True).values
    x = []
    y = []
    for i in range(100, 1001, 50):
        x.append(i)
        y.append(round(value[[(xi <= i) for xi in key]].sum()/3770 * 100, 2))

    df = pd.DataFrame({'Length': x, 'Percent distribution': y})

    fig = px.bar(df, x='Length', y='Percent distribution', title='Length Distribution',
                 text='Percent distribution')  # Use a bar chart for better visualization
    fig.update_traces(marker_color='red')
    # Customize labels and appearance
    fig.update_layout(
        xaxis_title=r'$ \leq $ Length',  # Use LaTeX for math notation
        yaxis_title='Percent (%)'
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig)

# ### Lottie test ###
def load_lottiefile(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
astronaut = load_lottiefile("/home/nhatdm2k4/Pictures/Data_mining/demo/resource/vietnamese_flag.json")
with st.sidebar:
    st.markdown("## ")
    st_lottie(astronaut, loop=True)
### Demo Section ###
#### Data for visualization ####
# trainloader =  joblib.load('/home/k64t/person-reid/demo_log/MiscStuff/btl_datamining/data/trainloader_300_drop')
# testloader = joblib.load('/home/k64t/person-reid/demo_log/MiscStuff/btl_datamining/data/testloader_300_drop')

proba = 0
if test_input:
    st.markdown('<h3 class="inknut-text margin-heading">Test Input</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    text_input = col1.text_area("", placeholder="Enter your text here", height=400)
    # Display the entered text
    predict = col1.button("Predict")
    new_text = convert2segment(text_input)
    embed_text = embedding_300(new_text)
    proba = inference(embed_text)
    if predict:
        if proba > 0.7:
            with col2:
                st.markdown(
                    f'<div style="font-size: 24px; text-align: center; margin-top:80px; font-size:50px; color: red">Phản động</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="font-size: 36px; text-align: center; margin-top:35px">{proba * 100:.2f}%</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="font-size: 16px; text-align: center; color: red"; margin-top:35px>{0.5 - proba}</div>',
                    unsafe_allow_html=True)

        else:
            with col2:
                st.markdown(
                    f'<div style="font-size: 24px; text-align: center; margin-top:80px; font-size:50px; color: green">Bình thường</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="font-size: 36px; text-align: center; margin-top:35px">{proba * 100:.2f}%</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="font-size: 16px; text-align: center; color: green"; margin-top:35px>{0.5 - proba}</div>',
                    unsafe_allow_html=True)

if performance_metrics:
    st.markdown('<h3 class="inknut-text margin-heading">Performance metrics</h3>', unsafe_allow_html=True)
@st.cache_data
def plot_tsne(df_tsne):
    chart = px.scatter_3d(df_tsne, x='A', y='B', z='C', color='label', color_continuous_scale='rainbow')
    return chart
@st.cache_data
def load_train_tsne():
    df_train_tsne = pd.read_csv('/home/nhatdm2k4/Pictures/Data_mining/demo/data/train_tsne.csv')
    return df_train_tsne
df_train_tsne = load_train_tsne()

@st.cache_data
def load_test_tsne():
    df_test_tsne = pd.read_csv('/home/nhatdm2k4/Pictures/Data_mining/demo/data/test_tsne.csv')
    return df_test_tsne
df_test_tsne = load_test_tsne()


if tsne_visualization:
    st.markdown('<h3 class="inknut-text margin-heading">T-sne visualization</h3>', unsafe_allow_html=True)
    fig_tsne_train = plot_tsne(df_train_tsne)
    fig_tsne_test = plot_tsne(df_test_tsne)
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_tsne_train)
    col2.plotly_chart(fig_tsne_test)

@st.cache_data
def load_train_pca():
    df_train_pca = pd.read_csv('/home/nhatdm2k4/Pictures/Data_mining/demo/data/train_pca.csv')
    return df_train_pca
df_train_pca = load_train_pca()

@st.cache_data
def load_test_pca():
    df_test_pca = pd.read_csv('/home/nhatdm2k4/Pictures/Data_mining/demo/data/test_pca.csv')
    return df_test_pca
df_test_pca = load_test_pca()

if pca_visualization:
    st.markdown('<h3 class="inknut-text margin-heading">PCA visualization</h3>', unsafe_allow_html=True)

    fig_pca_train = px.scatter_3d(df_train_pca, x='A', y='B', z='C', color='label', color_continuous_scale='rainbow')
    fig_pca_test = px.scatter_3d(df_test_pca, x='A', y='B', z='C', color='label', color_continuous_scale='rainbow')
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_pca_train)
    col2.plotly_chart(fig_pca_test)

if footer_state:
    footer = """
<style>
.footer {
    margin-top: 30px;
  width: 100%;
  color: black;
  text-align: center;
  height: 30px;
}

</style>
<div class="footer">

<p class="inknut-text" style="font-size:20px">Developed with ❤ by</p>
<div class="flex-container">
<p class='center-link' style="font-size:15px">Do Minh Nhat</p>
<p class='center-link' style="font-size:15px">Nguyen Quy Dang</p>
<p class='center-link' style="font-size:15px">Vu Van Long</p>
<p class='center-link' style="font-size:15px">Nguyen Nhat Minh</p>
</div>
"""
    st.markdown(footer, unsafe_allow_html=True)










