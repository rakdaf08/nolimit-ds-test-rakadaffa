"""
streamlit_app.py — RAG Chatbot UI
Run with: streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

import streamlit as st

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline import RAGPipeline

# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="📄 PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Multi-Document PDF RAG Assistant")
st.caption("Ask questions and get answers grounded in your document collection.")

# ------------------------------------------------------------------ #
#  Sidebar — configuration                                             #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.header("⚙️ Settings")

    # Detect Kaggle environment to set absolute paths as defaults
    is_kaggle = os.path.exists("/kaggle/working")
    kaggle_project_root = "/kaggle/working/nolimit-ds-test-rakadaffa"
    
    if is_kaggle and os.path.exists(kaggle_project_root):
        default_pdf_dir = os.path.join(kaggle_project_root, "data/pdfs")
        default_index_dir = os.path.join(kaggle_project_root, "faiss_index")
    else:
        default_pdf_dir = "data/pdfs"
        default_index_dir = "faiss_index"

    pdf_dir   = st.text_input("PDF Directory", value=default_pdf_dir)
    index_dir = st.text_input("Index Directory", value=default_index_dir)
    top_k     = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=5)

    st.divider()

    if st.button("🔄 Build / Reload Index", use_container_width=True):
        with st.spinner("Building index from PDFs..."):
            try:
                pipeline_obj = RAGPipeline(
                    pdf_dir=pdf_dir,
                    index_dir=index_dir,
                    top_k=top_k,
                )
                pipeline_obj.index(force_reindex=True)
                st.session_state["pipeline"] = pipeline_obj
                st.success("Index built successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("📂 Load Existing Index", use_container_width=True):
        with st.spinner("Loading saved index..."):
            try:
                pipeline_obj = RAGPipeline(
                    pdf_dir=pdf_dir,
                    index_dir=index_dir,
                    top_k=top_k,
                )
                pipeline_obj.load_index()
                st.session_state["pipeline"] = pipeline_obj
                st.success("Index loaded!")
            except Exception as e:
                st.error(f"Error: {e}")

    # Status indicator
    if "pipeline" in st.session_state and st.session_state["pipeline"].vector_store:
        n = st.session_state["pipeline"].vector_store.index.ntotal
        st.info(f"✅ Index ready — {n} vectors")
    else:
        st.warning("⚠️ No index loaded")

# ------------------------------------------------------------------ #
#  Chat history                                                        #
# ------------------------------------------------------------------ #

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------ #
#  Input                                                               #
# ------------------------------------------------------------------ #

question = st.chat_input("Ask a question about your documents...")

if question:
    if "pipeline" not in st.session_state:
        st.warning("Please build or load the index first (sidebar).")
        st.stop()

    # Display user message
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state["pipeline"].query(question)

                # Answer
                st.markdown(result["answer"])

                # Sources
                with st.expander("📚 Retrieved Sources", expanded=False):
                    for src in result["sources"]:
                        st.markdown(
                            f"**[{src['source_num']}] {src['filename']}** — "
                            f"Page {src['page_number']} | "
                            f"Score: `{src['score']:.4f}`"
                        )
                        st.caption(src["snippet"])
                        st.divider()

                st.session_state["messages"].append(
                    {"role": "assistant", "content": result["answer"]}
                )

            except Exception as e:
                st.error(f"Error generating answer: {e}")
