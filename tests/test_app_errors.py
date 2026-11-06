"""A failing embeddings call must show a redacted message in the chat, not crash."""

import app


def test_embedding_failure_is_shown_redacted_and_the_file_is_not_counted(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-secret")
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    def failing_add(chunks, filename):
        raise RuntimeError("401 Incorrect API key provided: sk-proj-abcdef123456 / sk-or-test-secret")

    monkeypatch.setattr(app.pipeline, "add_documents", failing_add)
    monkeypatch.setattr(app, "generate_summary_streaming", lambda docs, name, key="": iter(["summary"]))
    app.pipeline.reset()

    *_, last = app.process_and_respond("", [], [str(path)], "")
    text = str(last[1])

    assert "note.txt" in text and "401" in text
    assert "abcdef123456" not in text and "test-secret" not in text
    assert app.pipeline.processed_files == []
