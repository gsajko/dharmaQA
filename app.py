import streamlit as st
from pydantic.v1.error_wrappers import ValidationError

from utils import load_chain

# Configure streamlit page
st.set_page_config(page_title="Your Dharma Chatbot")
# Add header
st.title("Your Dharma Chatbot")

# Initialize LLM chain
chain, emb_repo, llm_repo = load_chain()

# side bar
st.sidebar.title("About")
st.sidebar.info("This is a demo of a chatbot that answers questions about the dharma.")
st.sidebar.write("Embedding model:")
st.sidebar.code(f"{emb_repo}", language="text")
st.sidebar.write("LLM model:")
st.sidebar.code(f"{llm_repo}", language="markdown")

# Chat logic
query = st.text_input("Ask me anything about Dharma", "How to practice attention?")
if query:
    # Send user's question to our chain
    try:
        result = chain.invoke(query)
    except ValidationError:
        result = "I don't understand your question."

    response = result["answer"]
    sources = []
    for source in result["context"]:
        sources.append(source.page_content)
    # sources = result["context"]

    # Display user question and assistant response
    st.write(f"User: {query}")
    st.write(f"Answer: {response}")

    # Display breakline
    st.write("---")
    
    for i, source in enumerate(sources):
        st.write(f"Context {i+1}:")
        st.code(f"{source}", language="text")
        # st.write(f"{source.page_content.metadata['title']}")
    # for k in result["context"][0].metadata:
    #     st.write(f"{k}")
    # TODO add metadata source title
