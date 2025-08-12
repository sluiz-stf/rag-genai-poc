import requests, streamlit as st

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG Demo com Re-ranking")

q = st.text_input("Pergunta")
if st.button("Perguntar") and q:
    resp = requests.post("http://localhost:8000/ask", json={"question": q}).json()
    st.write(resp["answer"])
    with st.expander("Fontes"):
        st.json(resp["sources"])