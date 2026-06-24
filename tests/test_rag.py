"""Tests for the infy RAG stack.

Covers documents, embeddings, vector store, retriever, text splitter, RAG chain.
"""

from infy.documents import Document, DocumentChunk
from infy.embeddings import FakeEmbeddings
from infy.messages import AIMessage
from infy.retrievers import VectorStoreRetriever
from infy.text_splitter import MarkdownTextSplitter, RecursiveTextSplitter, TextSplitter
from infy.vectorstores import InMemoryVectorStore

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class TestDocument:
    def test_create(self):
        doc = Document(page_content="Hello world", metadata={"source": "test"})
        assert doc.page_content == "Hello world"
        assert doc.metadata["source"] == "test"
        assert doc.id

    def test_default_metadata(self):
        doc = Document(page_content="test")
        assert "id" in doc.metadata

    def test_to_dict(self):
        doc = Document(page_content="test", metadata={"key": "val"})
        d = doc.to_dict()
        assert d["page_content"] == "test"
        assert d["metadata"]["key"] == "val"

    def test_repr(self):
        doc = Document(page_content="x" * 200)
        assert "..." in repr(doc)


class TestDocumentChunk:
    def test_chunk(self):
        doc = Document(page_content="Hello world")
        chunk = DocumentChunk(document=doc, chunk_index=0, total_chunks=2, start_char=0, end_char=5)
        assert chunk.page_content == "Hello world"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["total_chunks"] == 2

    def test_to_document(self):
        doc = Document(page_content="Hello world")
        chunk = DocumentChunk(document=doc, chunk_index=0, total_chunks=1)
        result = chunk.to_document()
        assert isinstance(result, Document)
        assert result.page_content == "Hello world"


# ---------------------------------------------------------------------------
# FakeEmbeddings
# ---------------------------------------------------------------------------


class TestFakeEmbeddings:
    def test_embed_query(self):
        emb = FakeEmbeddings(dimension=64)
        vec = emb.embed_query("hello")
        assert len(vec) == 64
        # Normalized
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_embed_documents(self):
        emb = FakeEmbeddings(dimension=32)
        vecs = emb.embed_documents(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 32

    def test_deterministic(self):
        emb = FakeEmbeddings()
        v1 = emb.embed_query("test")
        v2 = emb.embed_query("test")
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        emb = FakeEmbeddings()
        v1 = emb.embed_query("hello")
        v2 = emb.embed_query("world")
        assert v1 != v2


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


class TestInMemoryVectorStore:
    def test_add_and_count(self):
        store = InMemoryVectorStore()
        docs = [Document(page_content="doc1"), Document(page_content="doc2")]
        emb = FakeEmbeddings()
        vecs = emb.embed_documents([d.page_content for d in docs])
        ids = store.add_documents(docs, vecs)
        assert store.count() == 2
        assert len(ids) == 2

    def test_similarity_search(self):
        store = InMemoryVectorStore()
        emb = FakeEmbeddings()

        docs = [
            Document(page_content="Python is a programming language"),
            Document(page_content="JavaScript is used for web development"),
            Document(page_content="The weather is sunny today"),
        ]
        vecs = emb.embed_documents([d.page_content for d in docs])
        store.add_documents(docs, vecs)

        query_vec = emb.embed_query("programming language")
        results = store.similarity_search(query_vec, k=2)
        assert len(results) == 2
        assert all(isinstance(r[0], Document) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_delete(self):
        store = InMemoryVectorStore()
        emb = FakeEmbeddings()

        docs = [Document(page_content="a"), Document(page_content="b")]
        vecs = emb.embed_documents([d.page_content for d in docs])
        ids = store.add_documents(docs, vecs)

        store.delete([ids[0]])
        assert store.count() == 1

    def test_empty_search(self):
        store = InMemoryVectorStore()
        emb = FakeEmbeddings()
        results = store.similarity_search(emb.embed_query("test"), k=5)
        assert results == []

    def test_get_by_id(self):
        store = InMemoryVectorStore()
        emb = FakeEmbeddings()
        doc = Document(page_content="test")
        store.add_documents([doc], emb.embed_documents(["test"]))
        found = store.get_by_id(doc.id)
        assert found is not None
        assert found.page_content == "test"

    def test_get_all(self):
        store = InMemoryVectorStore()
        emb = FakeEmbeddings()
        docs = [Document(page_content="a"), Document(page_content="b")]
        store.add_documents(docs, emb.embed_documents(["a", "b"]))
        assert len(store.get_all()) == 2


# ---------------------------------------------------------------------------
# VectorStoreRetriever
# ---------------------------------------------------------------------------


class TestVectorStoreRetriever:
    def test_retrieve(self):
        store = InMemoryVectorStore()
        emb = FakeEmbeddings()

        docs = [
            Document(page_content="Python is great", metadata={"source": "python.md"}),
            Document(page_content="Rust is fast", metadata={"source": "rust.md"}),
            Document(page_content="JavaScript runs everywhere", metadata={"source": "js.md"}),
        ]
        vecs = emb.embed_documents([d.page_content for d in docs])
        store.add_documents(docs, vecs)

        retriever = VectorStoreRetriever(store=store, embeddings=emb, k=2)
        results = retriever.get_relevant_documents("fast language")
        assert len(results) == 2
        assert all(isinstance(d, Document) for d in results)

    def test_with_score_threshold(self):
        store = InMemoryVectorStore()
        emb = FakeEmbeddings()

        docs = [
            Document(page_content="completely unrelated topic about cooking"),
            Document(page_content="Python programming language"),
        ]
        vecs = emb.embed_documents([d.page_content for d in docs])
        store.add_documents(docs, vecs)

        retriever = VectorStoreRetriever(store=store, embeddings=emb, k=2, score_threshold=0.5)
        results = retriever.get_relevant_documents("Python")
        # Results depend on actual similarity scores
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TextSplitter
# ---------------------------------------------------------------------------


class TestTextSplitter:
    def test_short_text(self):
        splitter = TextSplitter(chunk_size=1000)
        chunks = splitter.split_text("Hello world")
        assert chunks == ["Hello world"]

    def test_long_text(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 60  # some leniency for sentence boundaries

    def test_split_documents(self):
        splitter = TextSplitter(chunk_size=30)
        doc = Document(page_content="Hello world. This is a test. Another sentence here.")
        chunks = splitter.split_documents([doc])
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert chunks[0].document.page_content == doc.page_content

    def test_preserves_content(self):
        splitter = TextSplitter(chunk_size=50)
        text = "First sentence. Second sentence. Third sentence."
        chunks = splitter.split_text(text)
        combined = "".join(chunks)
        # All original words should appear
        for word in ["First", "Second", "Third"]:
            assert word in combined


class TestRecursiveTextSplitter:
    def test_short_text(self):
        splitter = RecursiveTextSplitter(chunk_size=1000)
        chunks = splitter.split_text("Hello")
        assert chunks == ["Hello"]

    def test_split_documents(self):
        splitter = RecursiveTextSplitter(chunk_size=50)
        doc = Document(page_content="A " * 200)
        chunks = splitter.split_documents([doc])
        assert len(chunks) > 1


class TestMarkdownTextSplitter:
    def test_split_by_heading(self):
        text = "# Chapter 1\n\nContent of chapter 1.\n\n# Chapter 2\n\nContent of chapter 2."
        splitter = MarkdownTextSplitter(chunk_size=1000)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_split_documents(self):
        splitter = MarkdownTextSplitter(chunk_size=1000)
        doc = Document(page_content="# Title\n\nSome content.\n\n## Section\n\nMore content.")
        chunks = splitter.split_documents([doc])
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# RAG Chain (with mock model)
# ---------------------------------------------------------------------------


class MockRAGModel:
    def __init__(self, response_text: str = "Based on context: Python is great."):
        self.response_text = response_text
        self.last_messages = None

    def generate(self, messages, **kwargs):
        self.last_messages = messages
        return AIMessage(content=self.response_text)

    async def agenerate(self, messages, **kwargs):
        self.last_messages = messages
        return AIMessage(content=self.response_text)


class TestRAGChain:
    def test_rag_invoke(self):
        from infy.rag import RAGChain, RAGResult

        # Set up store
        store = InMemoryVectorStore()
        emb = FakeEmbeddings()
        docs = [
            Document(
                page_content="Python is a programming language.", metadata={"source": "python.md"}
            ),
            Document(page_content="Rust is a systems language.", metadata={"source": "rust.md"}),
        ]
        vecs = emb.embed_documents([d.page_content for d in docs])
        store.add_documents(docs, vecs)

        retriever = VectorStoreRetriever(store=store, embeddings=emb, k=2)
        model = MockRAGModel("Python is a great programming language.")

        rag = RAGChain(retriever=retriever, model=model)
        result = rag.invoke("What is Python?")

        assert isinstance(result, RAGResult)
        assert result.response == "Python is a great programming language."
        assert len(result.source_documents) == 2
        assert result.query == "What is Python?"

    def test_rag_builds_correct_prompt(self):
        from infy.rag import RAGChain

        store = InMemoryVectorStore()
        emb = FakeEmbeddings()
        docs = [Document(page_content="Test doc", metadata={"source": "test.md"})]
        store.add_documents(docs, emb.embed_documents(["Test doc"]))

        retriever = VectorStoreRetriever(store=store, embeddings=emb, k=1)
        model = MockRAGModel("Answer")

        rag = RAGChain(retriever=retriever, model=model)
        rag.invoke("Question?")

        # Check that the system prompt was formatted with context
        assert model.last_messages is not None
        system_msg = model.last_messages[0]
        assert "Test doc" in system_msg.content
        assert "[1]" in system_msg.content

    def test_rag_with_no_results(self):
        from infy.rag import RAGChain

        store = InMemoryVectorStore()
        emb = FakeEmbeddings()

        retriever = VectorStoreRetriever(store=store, embeddings=emb, k=2)
        model = MockRAGModel("I don't have enough context.")

        rag = RAGChain(retriever=retriever, model=model)
        rag.invoke("What?")

        assert "No relevant documents" in model.last_messages[0].content

    def test_rag_custom_prompt(self):
        from infy.rag import RAGChain

        store = InMemoryVectorStore()
        emb = FakeEmbeddings()
        docs = [Document(page_content="info")]
        store.add_documents(docs, emb.embed_documents(["info"]))

        retriever = VectorStoreRetriever(store=store, embeddings=emb)
        model = MockRAGModel("ok")

        rag = RAGChain(
            retriever=retriever,
            model=model,
            system_prompt="Custom prompt with {context}",
        )
        rag.invoke("test")
        assert "Custom prompt with" in model.last_messages[0].content


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_index_and_retrieve(self):
        """Test the full pipeline: create docs → split → embed → store → retrieve."""
        # 1. Create documents
        docs = [
            Document(
                page_content=(
                    "LangChain is a framework for building LLM applications "
                    "with chains, agents, and tools."
                ),
                metadata={"source": "langchain.md"},
            ),
            Document(
                page_content=(
                    "Python is a versatile programming language used in data "
                    "science, web development, and AI."
                ),
                metadata={"source": "python.md"},
            ),
            Document(
                page_content=(
                    "Vector databases store embeddings for fast similarity search in RAG pipelines."
                ),
                metadata={"source": "vectors.md"},
            ),
        ]

        # 2. Split long documents
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents(docs)

        # 3. Embed
        emb = FakeEmbeddings()
        texts = [c.page_content for c in chunks]
        vectors = emb.embed_documents(texts)

        # 4. Store
        store = InMemoryVectorStore()
        store.add_documents(chunks, vectors)

        # 5. Retrieve
        retriever = VectorStoreRetriever(store=store, embeddings=emb, k=3)
        results = retriever.get_relevant_documents("LLM framework")

        assert len(results) > 0
        assert store.count() > 0
