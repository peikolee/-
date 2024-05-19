import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST  # Import IST from track_utils

# Load Model
# 从pkl文件中加载模型字典
pipeline_file = open("C:/backup/毕业设计/基于机器学习的社交媒体抑郁情绪分析系统/Sentiment-Analysis-and-Emotional-Intelligence-in-Social-Media-with-AI-Data/emotion_classifier_pipe.pkl", "rb")
model_dict = joblib.load(pipeline_file)
pipeline_file.close()

# 从模型字典中获取两个模型
pipe_lr = model_dict["logistic_regression"]
pipe_rf = model_dict["random_forest"]

# Function to use Random Forest model for prediction
def predict_emotions_rf(docx):
    results = pipe_rf.predict([docx])
    return results[0]

# Function to use Logistic Regression model for prediction 
def predict_emotions_lr(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def predict_emotions(docx):
    try:
        results = predict_emotions_rf(docx)
    except:
        results = predict_emotions_lr(docx)
    return results

def get_prediction_proba(docx):#获取情感预测概率
    try:
        results = pipe_rf.predict_proba([docx])
    except:
        results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}
#字典 将情感类别映射到对应的表情符号

# Main Application
def main():
    st.title("Emotion Analysis App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))  #将当前访问页面的详情添加到数据库中
        st.subheader("Emotion Detection in Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST)) #将情感分析的详情添加到数据库中
            prompt = ''  #初始化第一个空字符串变量
            with col2:  #在第二行显示结果
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_) #将情感预测的概率转换为DataFrame格式
                proba_df_clean = proba_df.T.reset_index()  #对DataFrame进行转置和重置索引操作
                proba_df_clean.columns = ["emotions", "probability"]
                emotionRate = -1
                emotionName = ''
                for value in proba_df_clean.values:
                    if value[1]>emotionRate:
                        emotionRate=value[1]
                        emotionName=value[0]

                if emotionName=='neutral':
                    prompt='You are currently in a relatively stable emotional state. It is hoped that you maintain a calm and steady life and encounter things that bring you joy and happiness.'
                elif emotionName=='joy':
                    prompt='It\'s great to see you in a joyful mood. Wishing you a smooth and happy life every day.'
                elif emotionRate>0.6:
                    prompt='I\'m sorry to see that you\'re currently experiencing depressive feelings. I encourage you to seek professional help at the appropriate time. I wish you a speedy recovery from depression.'
                else:
                    prompt='I\'m sorry to hear that you\'re experiencing negative emotions, but thankfully, it\'s not too severe. I hope you can adjust your mindset and I wish you a quick recovery from this difficult situation, and may you encounter more joyful things ahead.'
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)
            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
                st.write(prompt)


    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)

    else:
        add_page_visited_details("About", datetime.now(IST))

        st.write("Welcome to the Emotion Detection in Text App! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions in textual data.")

        st.subheader("Our Mission")

        st.write("At Emotion Detection in Text, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text. We believe that emotions play a crucial role in communication, and by uncovering these emotions, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.")

        st.subheader("How It Works")

        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text. These features are then fed into the trained model, which predicts the emotions associated with the input text. The app displays the detected emotions, along with a confidence score, providing you with valuable insights into the emotional content of your text.")

        st.subheader("Key Features:")

        st.markdown("##### 1. Real-time Emotion Detection")

        st.write("Our app offers real-time emotion detection, allowing you to instantly analyze the emotions expressed in any given text. Whether you're analyzing customer feedback, social media posts, or any other form of text, our app provides you with immediate insights into the emotions underlying the text.")

        st.markdown("##### 2. Confidence Score")

        st.write("Alongside the detected emotions, our app provides a confidence score, indicating the model's certainty in its predictions. This score helps you gauge the reliability of the emotion detection results and make more informed decisions based on the analysis.")

        st.markdown("##### 3. User-friendly Interface")

        st.write("We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text, view the results, and interpret the emotions detected. Whether you're a seasoned data scientist or someone with limited technical expertise, our app is accessible to all.")

        st.subheader("Applications")

        st.markdown("""
          The Emotion Detection in Text App has a wide range of applications across various industries and domains. Some common use cases include:
          - Social media sentiment analysis
          - Customer feedback analysis
          - Market research and consumer insights
          - Brand monitoring and reputation management
          - Content analysis and recommendation systems
          """)


if __name__ == '__main__':
    main()

