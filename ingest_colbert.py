# %%
from ragatouille import RAGPretrainedModel
import pandas as pd
import pickle

# %%
# %%

# def get_files_list(csv_file):
#     files_list = []
#     df = pd.read_csv(csv_file)

#     df.columns = df.columns.str.replace(" ", "_").str.lower()

#     df = df.drop(df.index[0])


#     # split the transcript_or_writing column into pdf name and create new column
#     # remove .pdf from pdf_name
#     df["name"] = df.transcript_or_writing.str.split("/").str[-1].str.replace(".pdf", "")

#     cols = [
#         "name",
#         "date",
#         "title_of_event",
#         "title_of_talk_or_writing",
#         "broad_topics",
#         "detailed_topics",
#         "length_of_recording",
#         "type_of_recording",
#     ]
#     for row in df[cols].iterrows():
#         file_name = f"{(row[1])['name']}.md"
#         files_list.append(file_name)
#     return files_list
# files_list = get_files_list("data/Rob_Burbea_Transcripts.2023-12-31.csv")
# # %%
# def create_collection(files_list):
#     print("❗️ why colbert is running this function??? ❗️")
#     docs_collection = []
#     for file in files_list:
#         try:
#             markdown_path = f"data/md_parts/{file}"
#             with open(markdown_path, "r") as f:
#                 full_document = f.read()
#             docs_collection.append(full_document)
#         except FileNotFoundError:
#             error_msg = f"File {file} not found."
#             print(error_msg)
#     return docs_collection

# md_collection = create_collection(files_list)
# # %%
# # pickle list
# import pickle
# with open('md_collection.pkl', 'wb') as f:
#     pickle.dump(md_collection, f)

# %%
# load pickle
with open('md_collection.pkl', 'rb') as f:
    test_collection = pickle.load(f)
# %%
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

RAG.index(
    collection=test_collection,
    index_name="dharma_colb",
    max_document_length=180,
    split_documents=True,
)
# %%
