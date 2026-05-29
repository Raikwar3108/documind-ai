import streamlit as st
import fitz
import numpy as np
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import re
from rank_bm25 import BM25Okapi
load_dotenv()

# with open('streamlit_css.txt', 'r') as file:
#     # Read the entire content into a string
#     css_content = file.read()

# st.markdown(css_content, unsafe_allow_html=True)

# Load External CSS
def load_css(file_name):

    with open(file_name) as f:

        st.markdown(

            f"<style>{f.read()}</style>",

            unsafe_allow_html=True
        )


# Apply CSS
load_css("style.css")



client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Streamlit UI
# -----------------------------


st.title("📘DocuMind AI –  By Prashant Raikwar")
st.caption("Hybrid Retrieval (BM25 + Cosine) • Semantic Intelligence • Explainable & Context Aware AI",)

st.set_page_config(page_title="Hybrid RAG System", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

uploaded_file = st.file_uploader( "Upload Your PDF", type="pdf")

## extract pdf text

def extract_pdf_text(pdf_file):
    pdf_file = fitz.open(stream=pdf_file.read(), filetype="pdf")

    full_text = ""

    for page in pdf_file:

        text = page.get_text("text")

        full_text += text
    # Fix hyphenated words
    full_text = re.sub(r'-\s+', '', full_text)

    # Preserve paragraph blocks
    full_text = re.sub(r'\n\s*\n', '<<PARA>>', full_text)

    # Remove noisy single line breaks
    full_text = re.sub(r'\n', ' ', full_text)

    # Restore paragraphs
    full_text = full_text.replace('<<PARA>>', '\n\n')

    # Clean extra spaces
    full_text = re.sub(r'\s+', ' ', full_text)

    return(full_text)


def chunk_text_then_clean(full_text, chunk_size=800,chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n",". "," "], 
                                                    length_function=len,
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap )
    chunks_new = text_splitter.split_text(full_text)

    chunk_text = [ chunk.replace("\n", " ").replace("- ", "").strip() for chunk in chunks_new]
  
    return chunk_text



def create_embeddings(chunk_text, embedding_model="text-embedding-3-small"):

    embedding_model_1 =  client.embeddings.create(input=chunk_text, model=embedding_model)

    chunk_embeddings = [item.embedding for item in embedding_model_1.data]
    chunk_embeddings = np.array(chunk_embeddings,  dtype=np.float32)

    return (chunk_embeddings)



#Setup Retriever


def normal_retreiver(query,  chunk_embeddings, chunk_text, top_k=3):

    query_embedding_response = client.embeddings.create(input=[query], model="text-embedding-3-small")

    query_embedding = np.array(query_embedding_response.data[0].embedding, dtype=np.float32)

    similarities = cosine_similarity([query_embedding], chunk_embeddings)

    top_indices = similarities[0].argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:

        results.append({ "score": float(similarities[0][idx]),"chunk": chunk_text[idx] })


    return results


#### To capture the semantic meaning + keyword match both, we are running this hybrid search ( where cosine score will help in getting the semantic meaning & BM25 helps in giving the keyword search)

def hybrid_retriever(query, chunk_text, chunk_embeddings, top_k=3, alpha=0.5):

    # -----------------------------------
    # BM25 Retrieval
    # -----------------------------------

    
    tokenized_chunks = [ chunk.lower().split()  for chunk in chunk_text]
    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)


    # -----------------------------------
    # Embedding Retrieval
    # -----------------------------------
    
    query_embedding_response = client.embeddings.create(input=[query], model="text-embedding-3-small")


    query_embedding = np.array(query_embedding_response.data[0].embedding, dtype=np.float32)


    embedding_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # -----------------------------------
    # Normalize Scores
    # -----------------------------------


    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)


    embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-8)

    # -----------------------------------
    # Combine Scores
    # -----------------------------------

    final_scores = (alpha * embedding_scores + (1 - alpha) * bm25_scores)


    # -----------------------------------
    # Get Top Results
    # -----------------------------------
    top_indices = np.argsort(final_scores)[-top_k:][::-1]


    results = []


    for idx in top_indices:

        results.append({

            "score": float(final_scores[idx]),

            "bm25_score": float(bm25_scores[idx]),

            "embedding_score": float(embedding_scores[idx]),

            "chunk": chunk_text[idx]})

    
    return results
    
### Generate Final Answer after feeding context to LLM:

def generate_answer(query, context, model_name = "gpt-4o-mini"):
    prompt = f"""
        Answer the question using the context below:

        Context: 
        {context}

        Question:
        {query}

        """
    
    reponse = client.chat.completions.create(
                    model= model_name,
                    messages=[ {"role":"user","content":prompt}]
         )
    
    return reponse.choices[0].message.content

# def main_function_to_check(pdf_file, query):

#     full_text = extract_pdf_text(pdf_file)

#     chunk_text = chunk_text_then_clean(full_text, chunk_size=800,chunk_overlap=150)

#     chunk_embeddings = create_embeddings(chunk_text, embedding_model="text-embedding-3-small")

#     if retreiver_type =="normal":
#        top_indices =  normal_retreiver(query,  chunk_embeddings, chunk_text, top_k=3)
    
#     elif retreiver_type =="hybrid":
#        top_indices = hybrid_retriever(query, chunk_text, chunk_embeddings, top_k=3, alpha=0.5)


#     return(top_indices)


# def main_function_answer_generation(pdf_file, query, generation_type):

#     full_text = extract_pdf_text(pdf_file)

#     chunk_text = chunk_text_then_clean(full_text, chunk_size=800,chunk_overlap=150)

#     chunk_embeddings = create_embeddings(chunk_text, embedding_model="text-embedding-3-small")

#     if generation_type =="normal":
#        top_indices =  normal_retreiver(query,  chunk_embeddings, chunk_text, top_k=3)
    
#     elif generation_type =="hybrid":
#        top_indices = hybrid_retriever(query, chunk_text, chunk_embeddings, top_k=3, alpha=0.5)
    
    
#     join_chunk = []
#     for item in top_indices:
#         join_chunk.append(item['chunk'])

    
#     context  = "\n".join(join_chunk)

#     answer = generate_answer(query, context)

#     return (answer)


def main_function_1(pdf_file):
    full_text = extract_pdf_text(pdf_file)

    chunk_text = chunk_text_then_clean(full_text, chunk_size=800,chunk_overlap=150)

    chunk_embeddings = create_embeddings(chunk_text, embedding_model="text-embedding-3-small")

    return(chunk_text, chunk_embeddings)


def main_function_2(query,chunk_text, chunk_embeddings,generation_type):
    if generation_type =="normal":
        results =  normal_retreiver(query,  chunk_embeddings, chunk_text, top_k=3)
    
    elif generation_type =="hybrid":
       results = hybrid_retriever(query, chunk_text, chunk_embeddings, top_k=3, alpha=0.5)

    join_chunk = []
    for item in results:
        join_chunk.append(item['chunk'])

    
    context  = "\n".join(join_chunk)

    answer = generate_answer(query, context)

    return (answer,results)


# Stream lit functionality
if uploaded_file is not None:
    with st.spinner("Creating embeddings and building retriever...":
       chunk_text, chunk_embeddings =  main_function_1(uploaded_file)
    
    st.success("PDF Processed Successfully")
    
    query = st.text_input("Ask a question from PDF")
        
    search_type = st.selectbox("Select Retriever Search Type", ["Normal (Cosine Similarity)", "Hybrid (Cosine + BM25)"],index=1, help="Choose how retrieval should work" )
    

    if query:
            
            
            if search_type == "Normal (Cosine Similarity)":
                search_type1 = 'normal'
                st.write("Running basic retrieval...")

            elif search_type == "Hybrid (Cosine + BM25)":
                search_type1 = 'hybrid'
                st.write("Running hybrid retrieval...")


            answer,results = main_function_2(query,chunk_text, chunk_embeddings,search_type1)


            st.subheader("Here is your Answer: ")
            st.write(answer)


            st.subheader("Top 3 Retriever Results with Match Score out of 1 (Retriever Quality)")



            if search_type == "Normal (Cosine Similarity)":

                for idx, item in enumerate(results):

                    st.markdown("---")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Result", f"Result {idx+1}")


                    with col2:
                        st.metric("Semantic Score",round(item["score"],2))


            elif search_type == "Hybrid (Cosine + BM25)":
                                    
                for idx, item in enumerate(results):

                    st.markdown("---")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Result", f"Result {idx+1}")

                    with col2:
                        st.metric("Hybrid Score", round(item["score"], 2))

                    with col3:
                        st.metric("BM25 Score", round(item["bm25_score"], 2))

                    with col4:
                        st.metric("Semantic Score",round(item["embedding_score"],2))


            
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Like"):
            st.session_state.feedback = "like"
            st.write("Thank your for your feedback. See you again !!!!")

    with col2:
        if st.button("👎 Dislike"):
            st.session_state.feedback = "dislike"
            st.write("Thank your for your feedback. We are working hard to give you better results next time !!!")








    

