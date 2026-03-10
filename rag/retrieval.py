import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

PROMPT_TEMPLATE = """You are a helpful assistant. Use the following context extracted from the document to answer the question accurately.
If the answer cannot be found in the context, say "I couldn't find relevant information in the document for this question."

Context:
{context}

Question: {question}

Answer:"""


def get_qa_chain(vectorstore: Chroma):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return {"chain": chain, "retriever": retriever}


def answer_question(chain_dict: dict, question: str) -> dict:
    chain = chain_dict["chain"]
    retriever = chain_dict["retriever"]
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": sources,
    }
