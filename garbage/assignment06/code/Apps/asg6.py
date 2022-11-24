import streamlit as st

def app(data):
    st.title("Assignment No.6")

    st.set_option('deprecation.showPyplotGlobalUse', False)

    def printf(url):
         st.markdown(f'<p style="color:#000;font:lucida;font-size:25px;">{url}</p>', unsafe_allow_html=True)

    operation = st.selectbox("Operation", ["Hierarchical clustering - AGNES & DIANA. Plot Dendrogram",'k-Means','k-Medoids (PAM)', 'DBSCAN'])
