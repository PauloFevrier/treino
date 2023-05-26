import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from senha import API_KEY
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

os.chdir(r"C:\Users\Anwar Hermuche\Desktop\Projetos Code\Tutorial_PDFAnalise")
os.environ["OPENAI_API_KEY"] = API_KEY

with st.sidebar:
    st.title("Analisador de PDFs")
    st.markdown("""
    ### Sobre
    Este é um analisador de PDFs feito utilizando o framework langchain.

    ### Como utilizar
    - Faça o upload do arquivo PDF
    - Faça uma pergunta sobre o arquivo
    """)
    add_vertical_space(5)
    st.write("Criado por [Anwar Hermuche](https://www.linkedin.com/in/anwarhermuche)")


def main():
    st.header("Analisador de PDFs")

    # Importando PDF
    pdf = st.file_uploader(label = "Faça o upload do seu PDF")
    

    if pdf is not None:
        # Lendo o PDF
        pdf_reader = PdfReader(pdf)

        # Extraindo conteúdo do PDF
        conteudo = ""
        for pagina in pdf_reader.pages:
            conteudo_pagina = pagina.extract_text()
            conteudo += conteudo_pagina

        # Criando os chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(conteudo)
        
        # Verificando a exitência do vetor semântico
        nome_pdf = pdf.name[:-4]
        if os.path.exists(f"{nome_pdf}.pkl"):
            with open(file = f"{nome_pdf}.pkl", mode = 'rb') as file:
                vectorstore = pickle.load(file)
        else:
            embedding = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(texts = chunks, embedding = embedding)
            with open(file = f"{nome_pdf}.pkl", mode = 'wb') as file:
                pickle.dump(vectorstore, file)
            
        # Recebendo pergunta
        pergunta = st.text_input(label = "Faça uma pergunta")

        if pergunta:
            paginas_semanticas = vectorstore.similarity_search(query = pergunta)

            llm = OpenAI(temperature = 0, model_name = "gpt-3.5-turbo")
            chain = load_qa_chain(llm = llm, chain_type = "stuff")
            resposta = chain.run(input_documents = paginas_semanticas, question = pergunta)

            st.write(resposta)




if __name__ == "__main__":
    main()