import os
import gradio as gr
from document_processor import load_document, chunk_documents, generate_summary_streaming
from rag_pipeline import RAGPipeline

# Single shared pipeline instance for the app session
pipeline = RAGPipeline()


# --- Chat history format ---
# Gradio 6 removed the old "tuples" history format: the Chatbot now sends and
# expects a list of {"role": ..., "content": ...} messages. The handlers below
# still work with (user, assistant) pairs, so these two helpers convert at the
# boundary between the UI and the handlers.

def _text(content):
    """Return the plain text of a message's content (a string or a list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else str(content)


def to_pairs(messages):
    """Convert Gradio messages into (user, assistant) pairs."""
    pairs = []
    for msg in messages or []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        content = _text(msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", ""))
        if role == "user":
            pairs.append((content, ""))
        elif pairs and pairs[-1][0] is not None and pairs[-1][1] == "":
            pairs[-1] = (pairs[-1][0], content)
        else:
            pairs.append((None, content))
    return pairs


def to_messages(pairs):
    """Convert (user, assistant) pairs back into Gradio messages."""
    messages = []
    for user, assistant in pairs:
        if user is not None:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    return messages


def process_and_respond(message, chat_history, files):
    """Gradio entry point: converts the history, then streams the handler's updates."""
    for text, pairs in _process_and_respond_pairs(message, to_pairs(chat_history), files):
        yield text, to_messages(pairs)


def _process_and_respond_pairs(message, chat_history, files):
    """
    Main handler for all user interactions.
    Handles file uploads, document processing, and chat responses.
    """
    global pipeline

    # --- Handle file uploads ---
    if files:
        file_paths = [f.name if hasattr(f, "name") else f for f in files]

        # Show uploaded files in chat
        files_display = "📎 **Uploaded Files:**\n\n"
        for fp in file_paths:
            filename = os.path.basename(fp)
            size_kb = os.path.getsize(fp) / 1024
            ext = filename.split(".")[-1].upper()
            files_display += f"📄 **{filename}** `{ext} · {size_kb:.1f}KB`\n\n"

        display_msg = files_display + ("\n" + message if message and message.strip() else "")
        chat_history.append((display_msg.strip(), ""))
        yield "", chat_history

        # Process each file
        processing_msg = "✅ **Documents processed successfully!**\n\n"
        chat_history.append((None, processing_msg))
        yield "", chat_history

        successfully_processed = []
        errors = []

        for fp in file_paths:
            filename = os.path.basename(fp)

            # Skip duplicates
            if filename in pipeline.processed_files:
                continue

            docs, err = load_document(fp)
            if err:
                errors.append(f"❌ {filename}: {err}")
                continue

            # Stream the summary into chat
            for summary in generate_summary_streaming(docs, filename):
                all_content = processing_msg
                for prev in successfully_processed:
                    all_content += f"📄 **{prev}** ✓\n\n"
                all_content += summary
                chat_history[-1] = (None, all_content)
                yield "", chat_history

            # Add to pipeline
            chunks = chunk_documents(docs)
            pipeline.add_documents(chunks, filename)
            successfully_processed.append(filename)

        # Final status message
        final = chat_history[-1][1]
        final += "\n---\n\n💬 **Ready! Ask me anything about your documents.**"
        if errors:
            final += "\n\n⚠️ **Errors:**\n" + "\n".join(errors)
        chat_history[-1] = (None, final)
        yield "", chat_history

        # If user also sent a text message alongside files, answer it now
        if message and message.strip() and pipeline.is_ready():
            chat_history.append((message, ""))
            yield "", chat_history
            try:
                for _, updated_history in pipeline.stream_response(message, chat_history):
                    yield "", updated_history
            except Exception as e:
                chat_history[-1] = (message, f"❌ Error: {str(e)}")
                yield "", chat_history

        return

    # --- Handle text-only messages ---
    if message and message.strip():
        chat_history.append((message, ""))
        yield "", chat_history

        try:
            if pipeline.is_ready():
                for _, updated_history in pipeline.stream_response(message, chat_history):
                    yield "", updated_history
            else:
                for _, updated_history in pipeline.stream_direct_response(message, chat_history):
                    yield "", updated_history
        except Exception as e:
            chat_history[-1] = (message, f"❌ Error: {str(e)}")
            yield "", chat_history


def reset_pipeline(chat_history):
    """Reset all documents and notify the user in chat."""
    pipeline.reset()
    chat_history = to_pairs(chat_history)
    chat_history.append((None, "🔄 Knowledge base reset. All documents cleared."))
    return to_messages(chat_history)


def refresh_file_list():
    return pipeline.get_processed_files()


# --- Gradio UI ---
with gr.Blocks(title="DocuChat AI") as demo:
    gr.Markdown("# 💬 DocuChat AI")
    gr.Markdown("Upload PDF, DOCX, or TXT files and chat with your documents using RAG.")

    chatbot = gr.Chatbot(
        label="Chat",
        height=500,
        buttons=["copy"],
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question, or upload files above...",
            scale=4,
            show_label=False
        )
        file_upload = gr.Files(
            label="📎",
            file_types=[".pdf", ".docx", ".txt"],
            file_count="multiple",
            scale=1
        )

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary", scale=2)
        clear_btn = gr.Button("Clear Chat", scale=1)
        reset_btn = gr.Button("Reset Knowledge Base", variant="stop", scale=2)

    with gr.Accordion("📚 Processed Files", open=False):
        files_display = gr.Textbox(label="Documents in memory", interactive=False, lines=5)
        refresh_btn = gr.Button("Refresh", size="sm")

    # Event bindings
    submit_btn.click(
        process_and_respond,
        inputs=[msg, chatbot, file_upload],
        outputs=[msg, chatbot]
    )
    msg.submit(
        process_and_respond,
        inputs=[msg, chatbot, file_upload],
        outputs=[msg, chatbot]
    )
    submit_btn.click(lambda: None, None, file_upload)
    msg.submit(lambda: None, None, file_upload)

    clear_btn.click(lambda: [], None, chatbot)
    clear_btn.click(lambda: "", None, msg)

    reset_btn.click(reset_pipeline, inputs=[chatbot], outputs=[chatbot])
    refresh_btn.click(refresh_file_list, outputs=files_display)
    demo.load(refresh_file_list, outputs=files_display)


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())