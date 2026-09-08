from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os


def load_pdf(uploaded_file):
    """
    Load a Streamlit-uploaded PDF safely using a temporary file.

    The temporary file is deleted after the PDF has been loaded.
    """

    temp_path = None

    try:
        # Create temporary PDF
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        return docs

    finally:
        # Always delete temporary file
        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except OSError as e:
                print(
                    f"[WARNING] Could not delete temporary file: {e}"
                )