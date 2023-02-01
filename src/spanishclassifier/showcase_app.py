import datetime
import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

from spanishclassifier.utils.filters import *  

st.sidebar.markdown("## Models loaded")


st.title('Sentiment Analys for Spanish Tweets!')
st.write('I use the Hugging Face Transformers library to clasify the sentiment \
    of tweets passed as input as postive, neutral or negative. \
    This app is built using [Streamlit](https://docs.streamlit.io/en/stable/getting_started.html).')

models = [
    "francisco-perez-sorrosal/distilbert-base-uncased-finetuned-with-spanish-tweets-clf",
    "francisco-perez-sorrosal/distilbert-base-multilingual-cased-finetuned-with-spanish-tweets-clf",
    "francisco-perez-sorrosal/dccuchile-distilbert-base-spanish-uncased-finetuned-with-spanish-tweets-clf",
    "francisco-perez-sorrosal/distilbert-base-uncased-finetuned-with-spanish-tweets-clf-cleaned-ds",
    "francisco-perez-sorrosal/distilbert-base-multilingual-cased-finetuned-with-spanish-tweets-clf-cleaned-ds",
    "francisco-perez-sorrosal/dccuchile-distilbert-base-spanish-uncased-finetuned-with-spanish-tweets-clf-cleaned-ds",
]

load_all_models = st.checkbox("Load all models?")

if "pipelines" not in st.session_state:
    st.session_state.pipelines = []
    for model in models:
        with st.spinner(f"Loading model {model}"):
             pipe = pipeline(
                'text-classification', 
                model=AutoModelForSequenceClassification.from_pretrained(model),
                tokenizer=AutoTokenizer.from_pretrained(model),
                return_all_scores=True)
        st.sidebar.subheader(pipe.model.config.name_or_path)
        st.sidebar.write(f"Tokenizer:\n{pipe.tokenizer}")
        st.sidebar.write(f"Model:\n{pipe.model.config}")
        st.session_state.pipelines.append(pipe)
        if not load_all_models:
            break
    st.session_state.last_updated = datetime.time(0,0)

def update_model(model_id,):
    st.session_state.pipelines = []
    if not load_all_models:
        pipe = pipeline(
            'text-classification', 
            model=AutoModelForSequenceClassification.from_pretrained(model_id),
            tokenizer=AutoTokenizer.from_pretrained(model_id),
            return_all_scores=True)
        st.sidebar.subheader(pipe.model.config.name_or_path)
        st.sidebar.write(f"Tokenizer:\n{pipe.tokenizer}")
        st.sidebar.write(f"Model:\n{pipe.model.config}")
        st.session_state.pipelines.append(pipe)
    else:
        for model in models:
            with st.spinner(f"Loading model {model}"):
                pipe = pipeline(
                    'text-classification', 
                    model=AutoModelForSequenceClassification.from_pretrained(model),
                    tokenizer=AutoTokenizer.from_pretrained(model),
                    return_all_scores=True)
                st.sidebar.subheader(pipe.model.config.name_or_path)
                st.sidebar.write(f"Tokenizer:\n{pipe.tokenizer}")
                st.sidebar.write(f"Model:\n{pipe.model.config}")                    
            st.session_state.pipelines.append(pipe)
    st.session_state.last_updated = datetime.datetime.now().time()

model_id = st.selectbox(f"Choose a model {load_all_models}", models)
st.button("Load model/s", on_click=update_model, args=(model_id,))
st.write('Last Updated = ', st.session_state.last_updated)



form = st.form(key='sentiment-form')
tweet_text = form.text_area('Enter your tweet text')
clean_input = form.checkbox("Clean input text?")
submit = form.form_submit_button('Submit')

if submit:
    if clean_input:
        tweet_text = remove_twitter_handles({"text": tweet_text})["text"]
        tweet_text = remove_urls({"text": tweet_text})["text"]
        tweet_text = clean_html({"text": tweet_text})["text"]
    st.write(f"Sending this tweet content to the model: {tweet_text}")

    for classifier in st.session_state.pipelines:
        st.subheader(f"Model\n{classifier.model.config.name_or_path}")
        result = classifier(tweet_text)
        st.json(result, expanded=False)
        predictions=result[0]
        label = "N/A"
        score = -1
        for p in predictions:
            if p['score'] > score:
                label = p['label']
                score = p['score']

        if label == 'P':
            st.success(f'{label} sentiment (score: {score})')
        elif label == "NEU":
            st.warning(f'{label} sentiment (score: {score})')
        else:
            st.error(f'{label} sentiment (score: {score})')
