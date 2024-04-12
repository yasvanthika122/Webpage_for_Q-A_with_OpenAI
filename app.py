import streamlit as st
# Check the correct import statement
from langchain.document_loaders import AsyncChromiumLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
import openai
import nest_asyncio

# Set up OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Your existing code for model setup and retrieval
# ...


# Define the Streamlit app
def main():
    st.title("Webpage Q&A with OpenAI")

    # Input field for multiple URLs
    urls = st.text_area("Enter the URLs of the webpages (one URL per line):", "")

    # Split user input into a list of URLs
    url_list = [url.strip() for url in urls.split('\n') if url.strip()]

    if st.button("Ask Questions"):
        for url in url_list:
            # Process the URL and retrieve documents
            loader = AsyncChromiumLoader([url])
            docs = loader.load()
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(docs)
            text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
            chunked_documents = text_splitter.split_documents(docs_transformed)
            db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
            retriever = db.as_retriever(k=1)

            # Instantiate ConversationBufferMemory
            memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

            # Load memory
            loaded_memory = RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | get_buffer_string)

            # Generate standalone question
            standalone_question = {
                "standalone_question": {
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: get_buffer_string(x["chat_history"]),
                } | CONDENSE_QUESTION_PROMPT | standalone_query_generation_llm,
            }

            # Retrieve documents
            retrieved_documents = {
                "docs": itemgetter("standalone_question") | retriever,
                "question": lambda x: x["standalone_question"],
            }

            # Construct inputs for the final prompt
            final_inputs = {
                "context": lambda x: _combine_documents(x["docs"]),
                "question": itemgetter("question"),
            }

            # Get the answer
            answer = {
                "answer": final_inputs | ANSWER_PROMPT | response_generation_llm,
                "question": itemgetter("question"),
                "context": final_inputs["context"],
            }

            # Combine all the steps in the chain
            final_chain = loaded_memory | standalone_question | retrieved_documents | answer

            # Ask question to the model
            question = st.text_input(f"Ask your question about {url}:")
            if st.button("Get Answer"):
                result = call_conversational_rag(question, final_chain, memory)
                st.text(f"Answer for {url}: {result['answer']}")

# Run the Streamlit app
if __name__ == "__main__":
    main()

