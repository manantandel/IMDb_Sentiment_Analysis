import streamlit as st
import gzip
from bs4 import BeautifulSoup
import requests
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

def scrape(url_rec):
    title_list = []
    rating_list = []
    text_list = []

    url = url_rec
    
    r = requests.get(url)

    bs = BeautifulSoup(r.content, 'html.parser')

    review_containers = bs.find_all('div', class_='review-container')

    for i, review in enumerate(review_containers):
        title = review.a.contents
        try:
            title = ''.join(title)
        except:
            title = "No Title Given"

        date = str(review.find("span", {"class": "review-date"}).contents)
        rating = str(review.findAll('span')[1].contents)

        if date == rating:
            rating = "No Rating Given"
        else:
            rating = review.findAll('span')[1].contents
            rating = ''.join(rating)

        text = review.find("div", {"class": "text" })
        text = text.text

        title_list.append(title)
        rating_list.append(rating)
        text_list.append(text)

    review_data = pd.DataFrame(
        {
            'Title': title_list,
            'Rating': rating_list,
            'Review': text_list,
        }
    )
    st.toast("Scraping Completed", icon="💻")
    return review_data



def logistic_regression(review):
    with open("lr_model.pkl", 'rb') as file:
        lr_model = pickle.load(file)
    lr_result = lr_model.predict(review)
    return lr_result


def decision_tree(review):
    with open("dt_model.pkl", 'rb') as file:
        dt_model = pickle.load(file)
    dt_result = dt_model.predict(review)
    return dt_result


def nb(review):
    with open("nb_model.pkl", 'rb') as file:
        nb_model = pickle.load(file)
    nb_result = nb_model.predict(review)
    return nb_result    


def knn(review):
    with open('knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    knn_result = knn_model.predict(review)
    return knn_result


def vectorizer(review, mode):
    if mode == 0:
        with open("tfidf_knn.pkl", 'rb') as file:
                tfidf_knn = pickle.load(file)
        tfidf_knn = tfidf_knn.transform([review])
        return tfidf_knn
    else:
        with open("tfidf.pkl", 'rb') as file:
            tfidf_out = pickle.load(file)
        tfidf_out = tfidf_out.transform([review])
        return tfidf_out


def all_model(review):
    tfidf_knn = vectorizer(review, 0)
    tfidf_out = vectorizer(review, 1)

    positive,negative = [],[]
    accuracy = {
        "KNN":73,
        "NB":76,
        "DT":75,
        "LR":87
    }

    knn_output = knn(tfidf_knn)
    nb_output = nb(tfidf_out)
    logistic_regression_output = logistic_regression(tfidf_out)
    decision_tree_output = decision_tree(tfidf_out)

    knn_output = str(knn_output).lstrip("['")
    knn_output = knn_output.rstrip("']")
    if knn_output == "Positive":positive.append("KNN")
    else:negative.append("KNN")

    nb_output = str(nb_output).lstrip("['")
    nb_output = nb_output.rstrip("']")
    if nb_output == "Positive":positive.append("NB")
    else:negative.append("NB")

    logistic_regression_output = str(logistic_regression_output).lstrip("['")
    logistic_regression_output = logistic_regression_output.rstrip("']")
    if logistic_regression_output == "Positive":positive.append("LR")
    else:negative.append("LR")
    
    decision_tree_output = str(decision_tree_output).lstrip("['")
    decision_tree_output = decision_tree_output.rstrip("']")
    if decision_tree_output == "Positive":positive.append("LR")
    else:negative.append("DT")

    if len(positive) > len(negative): final_result = "Positive"
    elif len(positive) == len(negative):
        postive_acc = accuracy[positive[0]] + accuracy[positive[1]]
        negative_acc = accuracy[negative[0]] + accuracy[negative[1]]
        if postive_acc > negative_acc:
            final_result = "Positive"
        else: final_result = "Negative"
    else: final_result = "Negative"

    return knn_output,nb_output,logistic_regression_output,decision_tree_output, final_result


def get_movie(url):
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    review_elements = soup.find("h3")
    
    title = review_elements.find('a').text
    
    return title


def rating_sentiment(rating):
    if int(rating) >=6:
        return "Positive"
    else:
        return "Negative"
    

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎥",
    layout="wide",
)

st.write("""
    <div style="display: flex; justify-content: center;">
        <img src="https://i.imgur.com/q5IiPXi.png" alt="Centered Image">
    </div>
""", unsafe_allow_html=True)


new_title = '<p style="text-align: center;font-family:Impact; color:#e3b81e; font-size: 70px;">Sentiment Analysis</p>'
st.markdown(new_title, unsafe_allow_html=True)


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown(
    """<style>
    div[class*="row-widget stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 32px; color:#e3b81e;
    }
        </style>
        """, unsafe_allow_html=True)
analysis_choice = st.radio(label="Predict:", options=[":orange[User Input]",":orange[Enter URL]"])


if analysis_choice == ":orange[User Input]":
    st.markdown(
    """<style>
    div[class*="row-widget stTextInput st-emotion-cache-zn6u7j e11y4ecf0"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 32px; color:#e3b81e;
    }
        </style>
        """, unsafe_allow_html=True)
    model_choice = st.selectbox("Which Model do you want to use", ["All", "Naive Bayes", "Logistic Regression", "Decision Tree", "KNN"])



    st.markdown(
    """<style>
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 32px; color:#e3b81e;
    }
        </style>
        """, unsafe_allow_html=True)
    rating  = st.slider("Your Rating",1,10,6)



    st.markdown(
    """<style>
    div[class*="row-widget stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 32px; color:#e3b81e;
    }
        </style>
        """, unsafe_allow_html=True)
    review = st.text_input("Enter Review")

    if review:
        if model_choice == "All":
            rating_result = rating_sentiment(rating)
            knn_result,nb_result,logistic_regression_result,decision_tree_result,all_result = all_model(review)

            all_result = all_result and rating_result

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)
            with st.expander("Result: "):
                tab1, tab2, tab3, tab4, tab5= st.tabs(["Overall Sentiment", "KNN", "Naive Bayes", "Decision Tree", "Logistic Regression"])
                with tab1:
                    st.write(f"Rating: {rating}")
                    st.write(f"Review: {review}")
                    st.write(f"Overall Sentiment: {all_result}")
                    st.write(f"KNN Result: {knn_result}")
                    st.write(f"Naive Bayes Result: {nb_result}")
                    st.write(f"Decision Tree Result: {decision_tree_result}")
                    st.write(f"Logistic Regression Result: {logistic_regression_result}")
                
                with tab2:
                    st.write(f"Rating: {rating}")
                    st.write(f"Review: {review}")
                    st.write(f"KNN Result: {knn_result}")
                    st.write(f"KNN Accuracy: 73%")
                with tab3:
                    st.write(f"Rating: {rating}")
                    st.write(f"Review: {review}")
                    st.write(f"Naive Bayes Result: {nb_result}")
                    st.write(f"Naive Bayes Accuracy: 76%")
                with tab4:
                    st.write(f"Rating: {rating}")
                    st.write(f"Review: {review}")
                    st.write(f"Decision Tree Result: {decision_tree_result}")
                    st.write(f"Decision Tree Accuracy: 75%")
                with tab5:
                    st.write(f"Rating: {rating}")
                    st.write(f"Review: {review}")
                    st.write(f"Logistic Regression Result: {logistic_regression_result}")
                    st.write(f"Logistic Regression Accuracy: 87%")

        elif model_choice == "Naive Bayes":
            rating_result = rating_sentiment(rating)
            tfidf_review = vectorizer(review, 1)
            naive_out = nb(tfidf_review)
            nb_output = str(naive_out).lstrip("['")
            nb_output = nb_output.rstrip("']")
            final_output = nb_output and rating_result

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)
            with st.expander("Result: "):
                tab1 = st.tabs(["Naive Bayes"])
                st.write(f"Rating: {rating}")
                st.write(f"Review: {review}")
                st.write(f"Naive Bayes Result: {nb_output}")
                st.write(f"Star Rating Result: {rating_result}")
                st.write(f"Final Sentiment of Review: {final_output}")
                st.write(f"Naive Bayes Accuracy: 76%")
            
        elif model_choice == "Logistic Regression":
            rating_result = rating_sentiment(rating)
            tfidf_review = vectorizer(review, 1)
            lr_out = logistic_regression(tfidf_review)
            logistic_regression_output = str(lr_out).lstrip("['")
            logistic_regression_output = logistic_regression_output.rstrip("']")
            final_output = logistic_regression_output and rating_result

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)
            with st.expander("Result: "):
                tab1 = st.tabs(["Logistic Regression"])
                st.write(f"Rating: {rating}")
                st.write(f"Review: {review}")
                st.write(f"Logistic Regression Result: {logistic_regression_output}")
                st.write(f"Star Rating Result: {rating_result}")
                st.write(f"Final Sentiment Result: {final_output}")
                st.write(f"Logistic Regression Accuracy: 87%")
            
        elif model_choice == "Decision Tree":
            rating_result = rating_sentiment(rating)
            tfidf_review = vectorizer(review, 1)
            dt_out = decision_tree(tfidf_review)
            decision_tree_output = str(dt_out).lstrip("['")
            decision_tree_output = decision_tree_output.rstrip("']")
            final_output = decision_tree_output and rating_result

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)
            with st.expander("Result: "):
                tab1 = st.tabs(["Decision Tree"])
                st.write(f"Rating: {rating}")
                st.write(f"Review: {review}")
                st.write(f"Decision Tree Result: {decision_tree_output}")
                st.write(f"Star Rating Result: {rating_result}")
                st.write(f"Final Sentiment of Review: {final_output}")
                st.write(f"Decision Tree Accuracy: 75%")
            
        elif model_choice == "KNN":
            rating_result = rating_sentiment(rating)
            tfidf_review = vectorizer(review, 0)
            knn_out = knn(tfidf_review)
            knn_output = str(knn_out).lstrip("['")
            knn_output = knn_output.rstrip("']")
            final_output = knn_output and rating_result

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)
        
            with st.expander("Result: "):
                tab1 = st.tabs(["KNN"])
                st.write(f"Rating: {rating}")
                st.write(f"Review: {review}")
                st.write(f"KNN Result: {knn_output}")
                st.write(f"Star Rating Result: {rating_result}")
                st.write(f"Final Sentiment of Review: {final_output}")
                st.write(f"KNN Accuracy: 73%")
        

elif analysis_choice == ":orange[Enter URL]":
    st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)
    with st.expander("Which URL to Enter?"):
            st.write("Step 1: Open the IMDb Link of the Movie/Series.\nStep 2: Scroll Down to 'User Reviews' and Click on it\nStep 3: Copy the URL of the page which opens once you click 'User Review'")
            st.write("""
                <div style="display: flex; justify-content: center;">
                    <img src="https://i.imgur.com/DZSIkAK.png" alt="Centered Image">
                </div>
                """, unsafe_allow_html=True)
    review_url = st.text_input("Enter URL")
    st.markdown(
    """<style>
    div[class*="row-widget stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 32px; color:#e3b81e;
    }
    </style>
        """, unsafe_allow_html=True)
    model_choice = st.selectbox("Which Model do you want to use", ["All", "Naive Bayes", "Logistic Regression", "Decision Tree", "KNN"])
    
    st.markdown(
    """<style>
    div[class*="row-widget stTextInput st-emotion-cache-zn6u7j e11y4ecf0"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 32px; color:#e3b81e;
    }
        </style>
        """, unsafe_allow_html=True)
    start_button = st.button("Start Scraping")

    if start_button:
        df = scrape(review_url)
        movie_name = get_movie(review_url)

        df_process = df.copy()
        
        df_process["Rating"].replace("No Rating Given", np.nan, inplace=True)
        df_process['Rating'] = pd.to_numeric(df_process['Rating'], errors='coerce', downcast='integer')
        df_process["Rating"].fillna(df_process["Rating"].median(), inplace=True)
        review_list = df_process['Review'].to_list()
        rating_list = df_process["Rating"].to_list()
        df_ratings = df_process.groupby("Rating").size().reset_index(name="Count")

        if model_choice == "All":
            knn_list,nb_list,lr_list,dt_list,all_list = [],[],[],[],[]
            i = 0
            for review in review_list:
                rating_result = rating_sentiment(rating_list[i])
                knn_result,nb_result,logistic_regression_result,decision_tree_result,all_result = all_model(review)
                all_result = all_result and rating_result

                knn_list.append(knn_result)
                nb_list.append(nb_result)
                lr_list.append(logistic_regression_result)
                dt_list.append(decision_tree_result)
                all_list.append(all_result)
                i = i+1

            df_process["KNN"],df_process["Naive Bayes"], df_process['Logistic Regression'],df_process['Decision Tree'],df_process["Overall"] = knn_list, nb_list,lr_list, dt_list,all_list

            try:overall_positive = df_process['Overall'].value_counts()["Positive"]
            except:overall_positive = 0
            try:overall_negative = df_process['Overall'].value_counts()["Negative"]
            except:overall_negative = 0

            try:knn_positive = df_process['KNN'].value_counts()["Positive"]
            except:knn_positive = 0
            try:knn_negative = df_process['KNN'].value_counts()["Negative"]
            except:knn_negative = 0

            try:nb_positive = df_process['Naive Bayes'].value_counts()["Positive"]
            except:nb_positive = 0
            try:nb_negative = df_process['Naive Bayes'].value_counts()["Negative"]
            except:nb_negative = 0

            try:lr_positive = df_process['Logistic Regression'].value_counts()["Positive"]
            except:lr_positive = 0
            try:lr_negative = df_process['Logistic Regression'].value_counts()["Negative"]
            except:lr_negative = 0

            try:dt_positive = df_process['Decision Tree'].value_counts()["Positive"]
            except:dt_positive = 0
            try:dt_negative = df_process['Decision Tree'].value_counts()["Negative"]
            except:dt_negative = 0

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)

            with st.expander("Scraped Data"):
                tab1, tab2 = st.tabs(["Raw Data", "Analyzed Data"])
                with tab1:
                    st.dataframe(df)
                with tab2:
                    st.dataframe(df_process)

            with st.expander("Result"):
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overall Sentiment", "KNN", "Naive Bayes", "Decision Tree", "Logistic Regression"])
                with tab1:
                    st.write(f"Overall Sentiment: {df_process['Overall'].mode()[0]}")
                    st.write(f"Total Positive: {overall_positive}")
                    st.write(f"Total Negative: {overall_negative}")
        
                with tab2:
                    st.write(f"KNN Sentiment: {df_process['KNN'].mode()[0]}")
                    st.write(f"Total Positive: {knn_positive}")
                    st.write(f"Total Negative: {knn_negative}")
                    st.write(f"KNN Accuracy: 73%")
                with tab3:
                    st.write(f"Naive Bayes Sentiment: {df_process['Naive Bayes'].mode()[0]}")
                    st.write(f"Total Positive: {nb_positive}")
                    st.write(f"Total Negative: {nb_negative}")
                    st.write(f"Naive Bayes Accuracy: 76%")
                with tab4:
                    st.write(f"Decision Tree Sentiment: {df_process['Decision Tree'].mode()[0]}")
                    st.write(f"Total Positive: {dt_positive}")
                    st.write(f"Total Negative: {dt_negative}")
                    st.write(f"Decision Tree Accuracy: 75%")
                with tab5:
                    st.write(f"Logistic Regression Sentiment: {df_process['Logistic Regression'].mode()[0]}")
                    st.write(f"Total Positive: {lr_positive}")
                    st.write(f"Total Negative: {lr_negative}")
                    st.write(f"Logistic Regression Accuracy: 87%")

            with st.expander("Result Visualization"):
                tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs(["Overall Sentiment", "KNN", "Naive Bayes", "Decision Tree", "Logistic Regression","Rating Bar Graph"])
                with tab1:
                    fig = px.pie(title="Overall Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[overall_positive, overall_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig, use_container_width=True)
            
                with tab2:
                    fig = px.pie(title="KNN Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[knn_positive, knn_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab3:
                    fig = px.pie(title="Naive Bayes Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[nb_positive, nb_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab4:
                    fig = px.pie(title="Decision Tree Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[dt_positive, dt_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab5:
                    fig = px.pie(title="Logistic Regression Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[lr_positive, lr_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab6:
                    fig = px.bar(df_ratings,x="Rating",y="Count" ,title="Ratings", color_discrete_sequence=['#ffc600'])
                    fig.update_traces(width=0.7)
                    fig.update_layout(xaxis= dict(tickmode='linear'))
                    fig.update_xaxes(range=[1,10])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)

        elif model_choice == "KNN":
            knn_list,rating_sent,knn_overall = [],[],[]
            i = 0
            for review in review_list:
                rating_result = rating_sentiment(rating_list[i])
                tfidf_review = vectorizer(review, 0)
                knn_out = knn(tfidf_review)
                knn_output = str(knn_out).lstrip("['")
                knn_output = knn_output.rstrip("']")
                final_output = knn_output and rating_result
                knn_list.append(knn_output)
                rating_sent.append(rating_result)
                knn_overall.append(final_output)
                i+= 1
            df_process["KNN"] = knn_list
            df_process["Rating Sentiment"] = rating_sent
            df_process["Final"] = rating_sent

            try: knn_positive = df_process['KNN'].value_counts()["Positive"]
            except:knn_positive = 0
            try:knn_negative = df_process['KNN'].value_counts()["Negative"]
            except:knn_negative = 0

            try: final_positive = df_process['Final'].value_counts()["Positive"]
            except:final_positive = 0
            try:final_negative = df_process['Final'].value_counts()["Negative"]
            except:final_negative = 0

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)

            with st.expander("Scraped Data"):
                tab1, tab2 = st.tabs(["Raw Data", "Analyzed Data"])
                with tab1:
                    st.dataframe(df)
                with tab2:
                    st.dataframe(df_process)

            with st.expander("Result"):
                tab1,tab2 = st.tabs(["KNN", "Final"])
                with tab1:
                    st.write(f"KNN Sentiment: {df_process['KNN'].mode()[0]}")
                    st.write(f"Total Positive: {knn_positive}")
                    st.write(f"Total Negative: {knn_negative}")
                    st.write(f"KNN Accuracy: 73%")

                with tab2:
                    st.write(f"Final Sentiment: {df_process['Final'].mode()[0]}")
                    st.write(f"Ratings Sentiment: {df_process['Rating Sentiment'].mode()[0]}")
                    st.write(f"Total Positive: {final_positive}")
                    st.write(f"Total Negative: {final_negative}")

            with st.expander("Result Visualization"):
                tab1,tab2,tab3 = st.tabs(["KNN","Final" ,"Rating Bar Graph"])

                with tab1:
                    fig = px.pie(title="KNN Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[knn_positive, knn_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)

                with tab2:
                    fig = px.pie(title="Final Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[final_positive, final_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)

                with tab3:
                    fig = px.bar(df_ratings,x="Rating",y="Count" ,title="Ratings", color_discrete_sequence=['#ffc600'])
                    fig.update_traces(width=0.7)
                    fig.update_layout(xaxis= dict(tickmode='linear'))
                    fig.update_xaxes(range=[1,10])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)

        elif model_choice == "Naive Bayes":
            nb_list,rating_sent,nb_overall = [],[],[]
            i = 0
            for review in review_list:
                rating_result = rating_sentiment(rating_list[i])
                tfidf_review = vectorizer(review, 1)
                nb_out = nb(tfidf_review)
                nb_output = str(nb_out).lstrip("['")
                nb_output = nb_output.rstrip("']")
                final_output = nb_output and rating_result
                nb_list.append(nb_output)
                rating_sent.append(rating_result)
                nb_overall.append(final_output)
                i+=1
            df_process["Naive Bayes"] = nb_list
            df_process["Rating Sentiment"] = rating_sent
            df_process["Final"] = nb_overall

            try: nb_positive = df_process['Naive Bayes'].value_counts()["Positive"]
            except:nb_positive = 0
            try:nb_negative = df_process['Naive Bayes'].value_counts()["Negative"]
            except:nb_negative = 0
            try: final_positive = df_process['Final'].value_counts()["Positive"]
            except:final_positive = 0
            try:final_negative = df_process['Final'].value_counts()["Negative"]
            except:final_negative = 0

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)

            with st.expander("Scraped Data"):
                tab1, tab2 = st.tabs(["Raw Data", "Analyzed Data"])
                with tab1:
                    st.dataframe(df)
                with tab2:
                    st.dataframe(df_process)

            with st.expander("Result"):
                tab1,tab2 = st.tabs(["Naive Bayes", "Final"])
                with tab1:
                    st.write(f"Naive Bayes Sentiment: {df_process['Naive Bayes'].mode()[0]}")
                    st.write(f"Total Positive: {nb_positive}")
                    st.write(f"Total Negative: {nb_negative}")
                    st.write(f"Naive Bayes Accuracy: 76%")
                with tab2:
                    st.write(f"Naive Bayes Sentiment: {df_process['Naive Bayes'].mode()[0]}")
                    st.write(f"Ratings Sentiment: {df_process['Rating Sentiment'].mode()[0]}")
                    st.write(f"Total Positive: {final_positive}")
                    st.write(f"Total Negative: {final_negative}")

            with st.expander("Result Visualization"):
                tab1,tab2,tab3 = st.tabs(["Naive Bayes","Final" ,"Rating Bar Graph"])

                with tab1:
                    fig = px.pie(title="Naive Bayes Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[nb_positive, nb_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab2:
                    fig = px.pie(title="Final Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[final_positive, final_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab3:                    
                    fig = px.bar(df_ratings,x="Rating",y="Count" ,title="Ratings", color_discrete_sequence=['#ffc600'])
                    fig.update_traces(width=0.7)
                    fig.update_layout(xaxis= dict(tickmode='linear'))
                    fig.update_xaxes(range=[1,10])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)

        elif model_choice == "Decision Tree":
            dt_list,rating_sent,dt_overall = [],[],[]
            i=0
            for review in review_list:
                rating_result = rating_sentiment(rating_list[i])
                tfidf_review = vectorizer(review, 1)
                dt_out = decision_tree(tfidf_review)
                dt_output = str(dt_out).lstrip("['")
                dt_output = dt_output.rstrip("']")
                final_output = dt_output and rating_result
                dt_list.append(dt_output)
                rating_sent.append(rating_result)
                dt_overall.append(final_output)
                i+= 1
            df_process["Decision Tree"] = dt_list
            df_process["Rating Sentiment"] = rating_sent
            df_process["Final"] = dt_overall
            
            try: dt_positive = df_process['Decision Tree'].value_counts()["Positive"]
            except:dt_positive = 0
            try:dt_negative = df_process['Decision Tree'].value_counts()["Negative"]
            except:dt_negative = 0
            try: final_positive = df_process['Final'].value_counts()["Positive"]
            except:final_positive = 0
            try:final_negative = df_process['Final'].value_counts()["Negative"]
            except:final_negative = 0

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)

            with st.expander("Scraped Data"):
                tab1, tab2 = st.tabs(["Raw Data", "Analyzed Data"])
                with tab1:
                    st.dataframe(df)
                with tab2:
                    st.dataframe(df_process)

            with st.expander("Result"):
                tab1,tab2 = st.tabs(["Decision Tree", "Final"])
                with tab1:
                    st.write(f"Decision Tree Sentiment: {df_process['Decision Tree'].mode()[0]}")
                    st.write(f"Total Positive: {dt_positive}")
                    st.write(f"Total Negative: {dt_negative}")
                    st.write(f"Decision Tree Accuracy: 75%")
                with tab2:
                    st.write(f"Decision Tree Sentiment: {df_process['Final'].mode()[0]}")
                    st.write(f"Ratings Sentiment: {df_process['Rating Sentiment'].mode()[0]}")
                    st.write(f"Total Positive: {final_positive}")
                    st.write(f"Total Negative: {final_negative}")


            with st.expander("Result Visualization"):
                tab1,tab2,tab3 = st.tabs(["Decision Tree","Final" ,"Rating Bar Graph"])

                with tab1:
                    fig = px.pie(title="Decision Tree Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[dt_positive, dt_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab2:
                    fig = px.pie(title="Final Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[final_positive, final_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab3:                    
                    fig = px.bar(df_ratings,x="Rating",y="Count" ,title="Ratings", color_discrete_sequence=['#ffc600'])
                    fig.update_traces(width=0.7)
                    fig.update_layout(xaxis= dict(tickmode='linear'))
                    fig.update_xaxes(range=[1,10])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)

        elif model_choice == "Logistic Regression":
            lr_list,rating_sent,lr_overall = [],[],[]
            i = 0
            for review in review_list:
                rating_result = rating_sentiment(rating_list[i])
                tfidf_review = vectorizer(review, 1)
                lr_out = decision_tree(tfidf_review)
                lr_output = str(lr_out).lstrip("['")
                lr_output = lr_output.rstrip("']")
                final_output = lr_output and rating_result
                lr_list.append(lr_output)
                rating_sent.append(rating_result)
                lr_overall.append(final_output)

                i+=1
            df_process["Logistic Regression"] = lr_list
            df_process["Rating Sentiment"] = rating_sent
            df_process["Final"] = lr_overall
            try: lr_positive = df_process['Logistic Regression'].value_counts()["Positive"]
            except:lr_positive = 0
            try:lr_negative = df_process['Logistic Regression'].value_counts()["Negative"]
            except:lr_negative = 0
            try: final_positive = df_process['Final'].value_counts()["Positive"]
            except:final_positive = 0
            try:final_negative = df_process['Final'].value_counts()["Negative"]
            except:final_negative = 0

            st.markdown("""<style>
                        div[data-testid="stExpander"] div[role="button"] p {font-size: 26px;}</style>""", unsafe_allow_html=True)

            with st.expander("Scraped Data"):
                tab1, tab2 = st.tabs(["Raw Data", "Analyzed Data"])
                with tab1:
                    st.dataframe(df)
                with tab2:
                    st.dataframe(df_process)

            with st.expander("Result"):
                tab1,tab2 = st.tabs(["Logistic Regression", "Final"])
                with tab1:
                    st.write(f"Logistic Regression Sentiment: {df_process['Logistic Regression'].mode()[0]}")
                    st.write(f"Total Positive: {lr_positive}")
                    st.write(f"Total Negative: {lr_negative}")
                    st.write(f"Logistic Regression Accuracy: 87%")
                with tab2:
                    st.write(f"Logistic Regression Sentiment: {df_process['Final'].mode()[0]}")
                    st.write(f"Ratings Sentiment: {df_process['Rating Sentiment'].mode()[0]}")
                    st.write(f"Total Positive: {final_positive}")
                    st.write(f"Total Negative: {final_negative}")

            with st.expander("Result Visualization"):
                tab1,tab2,tab3 = st.tabs(["Logistic Regression","Final" ,"Rating Bar Graph"])

                with tab1:
                    fig = px.pie(title="Logistic Regression Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[lr_positive, lr_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab2:
                    fig = px.pie(title="Final Sentiment: Positive vs Negative" ,labels=["Positive", "Negative"], values=[final_positive, final_negative], names=["Positive", "Negative"], color_discrete_sequence=['#ffc600', '#CFE8EF'])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)
                with tab3:                    
                    fig = px.bar(df_ratings,x="Rating",y="Count" ,title="Ratings", color_discrete_sequence=['#ffc600'])
                    fig.update_traces(width=0.7)
                    fig.update_layout(xaxis= dict(tickmode='linear'))
                    fig.update_xaxes(range=[1,10])
                    st.plotly_chart(fig,theme="streamlit" ,use_container_width=True)

                