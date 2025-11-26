# Create Manual From Images

Streamlit app that converts ordered image sets into readable text summaries and PDF manuals using OpenAI or Azure OpenAI responses.

## Setup

```bash
pip install -r requirements.txt
```

Set your API key via environment variable or provide it in the sidebar when running the app:

```bash
export API_KEY=sk-...
# or for Azure OpenAI
export AZURE_OPENAI_API_KEY=xxxx
export AZURE_OPENAI_ENDPOINT=https://example-resource.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=my-gpt-4o-mini
export AZURE_OPENAI_API_VERSION=2024-02-01
streamlit run app.py
```

## Features

- Upload multiple images from the sidebar; files are auto-sorted by name.
- Choose OpenAI or Azure OpenAI in the UI, specify any available model/deployment (e.g., `gpt-4o-mini`, `gpt-4.1-mini`, Azure deployment name), and send images for descriptive Japanese text; outputs and raw API responses appear per image.
- Generate PDF manuals that embed the original uploaded images alongside their descriptions, plus downloadable JSON with local persistence.
- Download the finished manual as `.pdf` directly from the app.
