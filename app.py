from pathlib import Path
import os
import streamlit as st
# from pydantic.v1.error_wrappers import ValidationError

# from utils import load_chain

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
for item in (current_dir / "lancedb/dharma_qa.lance").iterdir():
    print(item)


# check if dataset is present
dataset_path = "mount/src/dharmaqa/lancedb/dharma_qa.lance"
if os.path.exists(dataset_path):
    print(f"The dataset exists at: {dataset_path}")
else:
    print(f"The dataset does not exist at: {dataset_path}")
# Initialize LLM chain
st.write(current_dir)
st.write("Hello world!")

for item in (current_dir / "lancedb/dharma_qa.lance/data").iterdir():
    st.write(item)

# for item in (current_dir / "lancedb/dharma_qa.lance/").iterdir():
#     st.write(item)