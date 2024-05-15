import streamlit as st

st.set_page_config(
    page_title="Charlotte (Chadaporn) Portfolio Homepage",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


if __name__ == "__main__":
    st.image('images/banner.png')
    st.title("🏡 Charlotte (Chadaporn) Portfolio")    
    st.subheader("🎯 Career Objective")
    st.write("A final year Data Science student (expecting to graduate in June 2024) with a background in Chemical Engineering and experience as a Procurement Analyst at PTT Global Chemical Public Company Limited (GC), Thailand's largest petrochemical company. Currently seeking a career in data science, data analytics, or business analysis, with a particular interest in utilising programming languages (Python), analytics tools (Alteryx), and data visualisation platforms (Tableau, Power BI, Google Looker Studio) to transform complex data into actionable insights and drive improved business outcomes.")
    
    st.subheader("📬 Personal Contact")
    st.write("email: charlotte.kchada@gmail.com")
    st.write("linkedin : https://linkedin.com/in/charlotte-chadaporn-khaolumloet-2807b920a")



