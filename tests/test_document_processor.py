"""document_processor tests. No API key or network needed."""

from langchain_core.documents import Document

from config import CHUNK_SIZE
from document_processor import chunk_documents, load_document


def test_long_text_is_split_into_chunks_within_the_size_limit():
    paragraphs = [f"Paragraph {i} " + "x" * 300 for i in range(10)]
    chunks = chunk_documents([Document(page_content="\n\n".join(paragraphs))])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
    joined = "\n".join(c.page_content for c in chunks)
    assert all(p in joined for p in paragraphs)  # nothing was dropped


def test_short_text_stays_in_one_chunk():
    chunks = chunk_documents([Document(page_content="Just one short paragraph.")])
    assert [c.page_content for c in chunks] == ["Just one short paragraph."]


def test_unsupported_file_type_returns_an_error():
    docs, err = load_document("notes.xyz")
    assert docs is None
    assert err == "Unsupported file type: xyz"


def test_txt_file_is_loaded_and_extension_is_case_insensitive(tmp_path):
    path = tmp_path / "NOTES.TXT"
    path.write_text("Hello DocuChat", encoding="utf-8")

    docs, err = load_document(str(path))

    assert err is None
    assert docs[0].page_content == "Hello DocuChat"


def test_missing_file_returns_an_error_instead_of_raising(tmp_path):
    docs, err = load_document(str(tmp_path / "missing.txt"))
    assert docs is None
    assert err.startswith("Error loading document")
