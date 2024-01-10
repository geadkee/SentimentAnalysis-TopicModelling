import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

st.set_page_config(
        page_title="Home",
        page_icon=":cityscape:",
        layout="centered", 
        initial_sidebar_state="expanded",
)

def run():
    
    # st.sidebar.markdown("# Home ")

    st.write("# **:grey[To Analyse the cause of]** Satisfaction **:grey[and]** Dissatisfaction **:grey[in]** Airbnb Accommodation **:grey[using]** Sentiment Analysis **:grey[and]** Topic Modelling")
    st.write("")
    st.write("")
    st.write("")
    # st.sidebar.success("Select Models :arrow_up:")

    st.markdown(
        """
        This research presents a methodology for investigating customers' satisfaction on accomodations based on online Airbnb listings' reviews, using
        **Hugging Face's Zero-shot Classification** (VADER) for sentiment analysis together with **BERTopic** & **Latent Dirichlet Allocation (LDA)** for topic modeling. 
        
        By identifying essential aspects and best practices, the approach aims **To examine the key attributes affecting tourists’ satisfaction and dissatisfaction towards accommodation through the analysis of Airbnb online reviews of Singapore, Thailand, and Turkey**.
        

        **👈 Select from the sidebar** to view the models!

        ### The objectives of this project
        - To find out the sentiments behind the tourists’ reviews on the Airbnb listings of Turkey, Singapore, and Thailand.
        - To generate topics from the tourists’ reviews on the Airbnb listings of Turkey, Singapore and Thailand.
        - To discover the source of satisfaction and dissatisfaction based on the attributes listed after topic modelling.
        - To offer practical suggestions for Malaysia's hospitality service based on the accommodation research.

        ### Check out the source code
        - My GitHuub [repository](https://github.com/geadkee/SentiAnalysis-TopicModel)
    """
    )


if __name__ == "__main__":
    run()