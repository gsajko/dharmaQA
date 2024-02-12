# dharmaQA

This is my "Hello World!" in the realm of RAGs.

This is a project to build a basic question answering system with RAG (Retrieval-Augmented Generation). Dataset used is dataset thats important to me, which is dataset made from Rob Burbea's Dharma talks.

Unfortunatelly, that **dataset isn't well-suited for RAG system** - it's not factual, it has long-winded answers, that are sometimes not directly related to the question.

For this kind of dataset, fine-tuning a language model would be more appropriate. 

I'll explore RAG using a different dataset, and then come back to this dataset later.


# Notes

App is deployed with [streamlit cloud](https://dharmaapp-u5sh7app6ruyzoy4zd93afy.streamlit.app/)

It retrieves context from transcripts of [Rob Burbea's Dharma talks](https://dharmaseed.org/teacher/210/), and generates a response based on the context.

Transcripts where downloaded from https://airtable.com/appe9WAZCVxfdGDnX/shr9OS6jqmWvWTG5g/tblHlCKWIIhZzEFMk/viw3k0IfSo0Dve9ZJ in the form of a pdf files.

I used [marker](https://github.com/VikParuchuri/marker) to convert pdf to Markdown files before ingesting them.