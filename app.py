from pathlib import Path

import streamlit as st
from pydantic.v1.error_wrappers import ValidationError

from utils import load_chain

# Configure streamlit page
st.set_page_config(page_title="Your Dharma Chatbot")


# get current working directory
current_dir = Path(__file__).parent.absolute()
print(current_dir)



# Print the contents of the current path
print("Contents:")
for item in current_dir.iterdir():
    print(item)
# display contents of lancedb directory
print("Contents of lancedb:")
for item in (current_dir / "lancedb").iterdir():
    print(item)


# Initialize LLM chain
chain = load_chain()

# Chat logic
query = st.text_input("Ask me anything")
if query:
    # Send user's question to our chain
    try:
        result = chain.invoke(query)
    except ValidationError:
        result = "I don't understand your question."

    response = result

    # Display user question and assistant response
    st.write(f"User: {query}")
    st.write(f"Assistant: {response}")
