import os
import streamlit as st
from rag_engine import NormasRAG, PDFDocumentLoader, VectorStoreManager

st.set_page_config(
    page_title="Sistema RAG - Consultas de Normas Institucionais",
    page_icon="📜",
    layout="wide"
)

# Initialize RAG engine in session state
@st.cache_resource
def get_rag_engine():
    return NormasRAG()

rag = get_rag_engine()

# Sidebar controls
st.sidebar.title("📜 RAG Normas")
st.sidebar.markdown("Consulta inteligente aos estatutos, regimentos e resoluções usando **Llama 3.2** via Ollama.")

# Check Ollama Status
ollama_online = rag.check_ollama_status()
if ollama_online:
    st.sidebar.success("Servidor Ollama: Online (llama3.2)")
else:
    st.sidebar.error("Servidor Ollama: Offline! Execute 'ollama serve' no terminal.")

# Index status
vector_mgr = rag.vector_store
doc_count = vector_mgr.collection.count()
st.sidebar.metric("Chunks Indexados", doc_count)

# Category options
categories = ["Todas", "consad", "estatuto", "plano_gestao", "regimento_geral", "regimento_interno_reitoria"]
selected_category = st.sidebar.selectbox("Filtrar Categoria", categories)

top_k = st.sidebar.slider("Trechos Recorridos (Top K)", min_value=1, max_value=10, value=5)

st.sidebar.divider()
if st.sidebar.button("Reconstruir Índice Vetorial"):
    with st.spinner("Indexando PDFs de /home/lucas/workspace/valhalla/documentos..."):
        loader = PDFDocumentLoader()
        chunks = loader.extract_chunks()
        vector_mgr.build_index(chunks, force=True)
        st.sidebar.success(f"Índice reconstruído! {len(chunks)} chunks.")
        st.rerun()

# Main Title & Chat Interface
st.title(" Consultor RAG de Normas & Resoluções")
st.markdown(
    "Faça perguntas sobre os documentos localizados em `/home/lucas/workspace/valhalla/documentos` "
    "(Estatuto, Plano de Gestão, Resoluções do CONSAD, etc.)."
)

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Ver fontes consultadas"):
                for idx, src in enumerate(msg["sources"], 1):
                    meta = src["metadata"]
                    st.markdown(
                        f"**[{idx}] {meta['file_name']}** — Página {meta['page_number']} "
                        f"*(Categoria: `{meta['category']}`, Relevância: {src['similarity']*100:.1f}%)*\n\n"
                        f"> {src['text']}"
                    )

# Input prompt
if prompt := st.chat_input("Ex: Qual o procedimento para criação de cursos novos no CONSAD?"):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Buscando trechos das normas e consultando o Llama 3.2..."):
            cat_filter = selected_category if selected_category != "Todas" else None
            res = rag.query(prompt, top_k=top_k, category_filter=cat_filter)
            
            st.markdown(res["answer"])
            
            if res["sources"]:
                with st.expander("Ver fontes consultadas"):
                    for idx, src in enumerate(res["sources"], 1):
                        meta = src["metadata"]
                        st.markdown(
                            f"**[{idx}] {meta['file_name']}** — Página {meta['page_number']} "
                            f"*(Categoria: `{meta['category']}`, Relevância: {src['similarity']*100:.1f}%)*\n\n"
                            f"> {src['text']}"
                        )
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "sources": res["sources"]
            })
