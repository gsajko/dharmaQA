# %%
import hashlib
import pickle

import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import LanceDB

# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

import lancedb

# %%
csv_file = "data/Rob_Burbea_Transcripts.2023-12-31.csv"
df = pd.read_csv(csv_file)

df.columns = df.columns.str.replace(" ", "_").str.lower()

df = df.drop(df.index[0])


# split the transcript_or_writing column into pdf name and create new column
# remove .pdf from pdf_name
df["name"] = df.transcript_or_writing.str.split("/").str[-1].str.replace(".pdf", "")

cols = [
    "name",
    "date",
    "title_of_event",
    "title_of_talk_or_writing",
    "broad_topics",
    "detailed_topics",
    "length_of_recording",
    "type_of_recording",
]


# %%
def process_documents(
    df, cols, start_row=0, text_splitter=None, md_path="data/md_parts/"
):
    docs = []
    not_processed = []
    total_rows = len(df[cols])

    for i, row in enumerate(df[cols][start_row:].iterrows(), start=start_row):
        fields_from_df = dict(row[1])
        markdown_path = f"{md_path}{fields_from_df['name']}.md"
        try:
            loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
            data = loader.load()
            text_chunks = [chunk.page_content for chunk in data]
            id = fields_from_df["name"]
            hash_value = hashlib.sha1(id.encode()).hexdigest()
            fields = {
                "id": hash_value,
                "chunks": text_chunks,
                "title_of_talk_or_writing": fields_from_df["title_of_talk_or_writing"],
            }
            try:
                for x in range(len(data)):
                    data[x].metadata = fields_from_df
            except IndexError:
                error_msg = f"File {markdown_path} is empty."
                print(error_msg)
                not_processed.append(error_msg)

        except FileNotFoundError:
            error_msg = f"File {markdown_path} not found."
            print(error_msg)
            not_processed.append(error_msg)
        print(f"Processing {i}/{total_rows} md files")
        docs.append(fields)

    print("Documents created ✨")
    return docs, not_processed


# %%
docs, not_processed = process_documents(df, cols)
# %%
# # show the first document structure
for k, v in docs[0].items():
    print(k, type(v))


# %%
# right now we have a list of chunks
# we need to separate them into individual documents
def separate_chunks(docs):
    separated_docs = []
    for doc in docs:
        for chunk in doc["chunks"]:
            separated_doc = {
                "id": doc["id"],
                "title_of_talk_or_writing": doc["title_of_talk_or_writing"],
                "text": chunk,
            }
            separated_docs.append(separated_doc)
    return separated_docs


separated_docs = separate_chunks(docs)


# %%
# to use the lancedb we need to transform dicts into Document objects
def transform_dicts_to_docs(docs):
    documents = []
    for item in range(len(docs)):
        page = Document(
            page_content=docs[item]["text"],
            metadata={"title": docs[item]["title_of_talk_or_writing"]},
        )
        documents.append(page)
    return documents


documents = transform_dicts_to_docs(separated_docs)
# %%
# save the documents to disk
with open("data/processed_docs.pickle", "wb") as f:
    pickle.dump(documents, f)
# %%
# load the documents from disk
with open("data/processed_docs.pickle", "rb") as f:
    documents = pickle.load(f)
# %%
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# %%
db = lancedb.connect("lancedb")

table = db.create_table(
    "dharma_qa",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            # "title": "Doc Title", TODO
            # https://lancedb.github.io/lancedb/notebooks/code_qa_bot/
            "id": "1",
        }
    ],
    mode="overwrite",
)
# %%

docsearch = LanceDB.from_documents(
    documents=documents, embedding=embeddings, connection=table
)
# %%


# %%
for k, v in documents[0]:
    print(k, type(v))
# %%
documents[0].metadata

# %%
# %%
