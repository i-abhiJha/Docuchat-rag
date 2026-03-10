import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    generation: str
    steps: List[str]
    route_decision: str       # "retrieve" or "direct_answer"
    retry_count: int
    hallucination_ok: bool
    hallucination_retries: int


# ── Structured output schemas ─────────────────────────────────────────────────

class RouteDecision(BaseModel):
    datasource: str = Field(description="Either 'retrieve' or 'direct_answer'")


class GradeDecision(BaseModel):
    score: str = Field(description="Either 'yes' or 'no'")


# ── Helper ────────────────────────────────────────────────────────────────────

def _llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Decide whether the question needs document retrieval or can be answered directly."""
    llm = _llm().with_structured_output(RouteDecision)
    prompt = PromptTemplate.from_template(
        "You are a query router. Decide if the question requires searching an uploaded document, "
        "or if it can be answered directly from general knowledge.\n\n"
        "Question: {question}\n\n"
        "Reply 'retrieve' if it asks about specific document content, facts, or details from an uploaded file. "
        "Reply 'direct_answer' if it is general knowledge, math, greetings, or small talk."
    )
    result: RouteDecision = (prompt | llm).invoke({"question": state["question"]})
    decision = result.datasource if result.datasource in ("retrieve", "direct_answer") else "retrieve"
    return {
        "route_decision": decision,
        "rewritten_question": state["question"],
        "steps": state.get("steps", []) + [f"Router: **{decision}**"],
    }


def grade_documents_node(state: AgentState) -> dict:
    """Keep only chunks that are relevant to the question."""
    llm = _llm().with_structured_output(GradeDecision)
    prompt = PromptTemplate.from_template(
        "Is the document chunk below relevant to answering the question?\n\n"
        "Question: {question}\n\nChunk: {document}\n\n"
        "Reply 'yes' if relevant, 'no' if not."
    )
    question = state.get("rewritten_question") or state["question"]
    relevant = []
    for doc in state.get("documents", []):
        try:
            result: GradeDecision = (prompt | llm).invoke(
                {"question": question, "document": doc.page_content}
            )
            if result.score.lower() == "yes":
                relevant.append(doc)
        except Exception:
            relevant.append(doc)  # keep on error

    return {
        "documents": relevant,
        "steps": state.get("steps", []) + [
            f"Grader: **{len(relevant)}/{len(state.get('documents', []))}** chunks relevant"
        ],
    }


def rewrite_query_node(state: AgentState) -> dict:
    """Rewrite the query to improve retrieval on the next attempt."""
    llm = _llm()
    prompt = PromptTemplate.from_template(
        "The query below did not retrieve useful document chunks. "
        "Rewrite it to be more specific and better match the document's content.\n\n"
        "Original query: {question}\n\nRewritten query:"
    )
    rewritten = (prompt | llm | StrOutputParser()).invoke({"question": state["question"]})
    rewritten = rewritten.strip()
    return {
        "rewritten_question": rewritten,
        "retry_count": state.get("retry_count", 0) + 1,
        "steps": state.get("steps", []) + [f"Query rewritten: *{rewritten}*"],
    }


def generate_node(state: AgentState) -> dict:
    """Generate the answer, using retrieved docs if available."""
    llm = _llm()
    docs = state.get("documents", [])

    if docs:
        prompt = PromptTemplate.from_template(
            "You are a helpful assistant. Use the context below to answer the question.\n"
            "If the answer is not in the context, say so clearly.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": _format_docs(docs), "question": state["question"]}
        )
    else:
        prompt = PromptTemplate.from_template(
            "Answer the following question using your general knowledge.\n\n"
            "Question: {question}\n\nAnswer:"
        )
        answer = (prompt | llm | StrOutputParser()).invoke({"question": state["question"]})

    return {
        "generation": answer,
        "steps": state.get("steps", []) + ["Answer generated"],
    }


def check_hallucination_node(state: AgentState) -> dict:
    """Verify the answer is grounded in the retrieved documents."""
    docs = state.get("documents", [])
    if not docs:
        return {
            "hallucination_ok": True,
            "steps": state.get("steps", []) + ["Hallucination check: skipped (no docs used)"],
        }

    llm = _llm().with_structured_output(GradeDecision)
    prompt = PromptTemplate.from_template(
        "Is the answer below fully grounded in and supported by the provided documents?\n\n"
        "Documents: {documents}\n\nAnswer: {generation}\n\n"
        "Reply 'yes' if every claim in the answer comes from the documents. "
        "Reply 'no' if the answer contains information not found in the documents."
    )
    try:
        result: GradeDecision = (prompt | llm).invoke(
            {"documents": _format_docs(docs), "generation": state["generation"]}
        )
        is_ok = result.score.lower() == "yes"
    except Exception:
        is_ok = True  # assume ok on error

    retries = state.get("hallucination_retries", 0)
    label = "passed" if is_ok else "failed"
    new_retries = retries if is_ok else retries + 1

    return {
        "hallucination_ok": is_ok,
        "hallucination_retries": new_retries,
        "steps": state.get("steps", []) + [f"Hallucination check: **{label}**"],
    }


# ── Conditional edges ─────────────────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    return "retrieve" if state.get("route_decision") == "retrieve" else "generate"


def route_after_grading(state: AgentState) -> str:
    if state.get("documents"):
        return "generate"
    if state.get("retry_count", 0) < 2:
        return "rewrite_query"
    return "generate"  # give up and answer with no context


def route_after_hallucination(state: AgentState) -> str:
    if state.get("hallucination_ok", True):
        return "end"
    if state.get("hallucination_retries", 0) < 2:
        return "regenerate"
    return "end"  # stop after 2 failed regeneration attempts


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent_graph(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def retrieve_node(state: AgentState) -> dict:
        """Fetch top-K chunks from ChromaDB."""
        query = state.get("rewritten_question") or state["question"]
        docs = retriever.invoke(query)
        return {
            "documents": docs,
            "steps": state.get("steps", []) + [f"Retrieved **{len(docs)}** chunks for: *{query}*"],
        }

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_hallucination", check_hallucination_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router", route_after_router,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents", route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination", route_after_hallucination,
        {"end": END, "regenerate": "generate"},
    )

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(agent, question: str) -> dict:
    initial_state: AgentState = {
        "question": question,
        "rewritten_question": question,
        "documents": [],
        "generation": "",
        "steps": [],
        "route_decision": "retrieve",
        "retry_count": 0,
        "hallucination_ok": True,
        "hallucination_retries": 0,
    }
    final_state = agent.invoke(initial_state)
    return {
        "answer": final_state["generation"],
        "sources": final_state.get("documents", []),
        "steps": final_state.get("steps", []),
    }
