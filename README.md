# Create Manual From Images

Streamlit app that converts ordered image sets into readable text summaries and PDF manuals using the OpenAI API.

## Setup

```bash
pip install -r requirements.txt
```

Set your API key via environment variable or provide it in the sidebar when running the app:

```bash
export API_KEY=sk-...
streamlit run app.py
```

## Features

- Upload multiple images from the sidebar; files are auto-sorted by name.
- Send images to OpenAI (`gpt-4o-mini`) for descriptive Japanese text; outputs and raw API responses appear per image.
- Generate PDF manuals that embed the original uploaded images alongside their descriptions, plus downloadable JSON with local persistence.
- Download the finished manual as `.pdf` directly from the app.
