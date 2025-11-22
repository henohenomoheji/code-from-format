import base64
import json
import logging
import os
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import streamlit as st
from openai import OpenAI

if TYPE_CHECKING:  # pragma: no cover - hints only
    from streamlit.runtime.uploaded_file_manager import UploadedFile


logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Create Manual From images",
    layout="wide",
    page_icon="📘",
)


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _create_batch_dir() -> Path:
    """Create (or reuse) a timestamped directory data/YYYYMMDDHHMM/."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    batch_dir = DATA_DIR / timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def _resolve_mime(file_name: str, provided: str | None) -> str:
    """Return a best-effort mime string for the given file."""
    if provided:
        return provided
    guessed, _ = mimetypes.guess_type(file_name)
    return guessed or "image/png"


def _persist_uploaded_files(files: List["UploadedFile"]) -> Dict[str, Dict[str, str]]:
    """Save uploaded files to disk and store their paths/mime in session state."""
    if not files:
        return {}

    batch_dir = _create_batch_dir()
    saved: Dict[str, Dict[str, str]] = {}

    for file in files:
        safe_name = Path(file.name).name
        destination = batch_dir / safe_name
        destination.write_bytes(file.getvalue())
        file.seek(0)
        saved[file.name] = {
            "path": str(destination),
            "mime": _resolve_mime(file.name, file.type),
        }

    st.session_state["saved_images"] = saved
    st.session_state["current_batch_dir"] = str(batch_dir)
    return saved


def _ensure_saved_images(files: List["UploadedFile"]) -> Dict[str, Dict[str, str]]:
    """Persist files when necessary and return the saved metadata."""
    if not files:
        st.session_state["saved_images"] = {}
        st.session_state["current_batch_dir"] = ""
        return {}

    saved: Dict[str, Dict[str, str]] = st.session_state.get("saved_images", {})
    uploaded_names = {file.name for file in files}
    saved_names = set(saved.keys())
    paths_exist = all(Path(meta["path"]).exists() for meta in saved.values()) if saved else False

    if not saved or uploaded_names != saved_names or not paths_exist:
        return _persist_uploaded_files(files)

    return saved


def _init_state() -> None:
    """Ensure session keys exist."""
    st.session_state.setdefault("ai_responses", {})
    st.session_state.setdefault("manual_json", {})
    st.session_state.setdefault("manual_markdown", "")
    st.session_state.setdefault("saved_json_path", "")
    st.session_state.setdefault("raw_texts", {})
    st.session_state.setdefault("saved_images", {})
    st.session_state.setdefault("current_batch_dir", "")


def describe_image(client: OpenAI, file_name: str, image_bytes: bytes) -> Dict[str, Any]:
    """Send the image to OpenAI Vision and return both the description and raw text blocks."""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "The user is creating a procedural manual. "
                                "Describe the key action, tools, and context shown in this image. "
                                "Write clear, imperative steps in Japanese."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
        logging.info(
            "OpenAI raw response for %s:\n%s",
            file_name,
            response.model_dump_json(indent=2),
        )
    except Exception as exc:  # pragma: no cover - surfacing errors to UI
        st.error(f"OpenAI API call failed: {exc}")
        raise

    text_blocks: List[str] = []
    if response.output:
        for block in response.output:
            for piece in block.content:
                if hasattr(piece, "text"):
                    text_blocks.append(piece.text)

    primary_text = text_blocks[0].strip() if text_blocks else ""
    return {"text": primary_text, "raw_blocks": text_blocks}


def build_manual_sections(
    files: List["UploadedFile"],
    responses: Dict[str, str],
    saved_assets: Dict[str, Dict[str, str]],
):
    """Prepare consistent metadata for downstream formatting using saved paths."""
    sections = []
    for file in files:
        asset = saved_assets.get(file.name)
        if not asset:
            continue
        asset_path = Path(asset["path"])
        if not asset_path.exists():
            continue
        sections.append(
            {
                "name": file.name,
                "mime": asset["mime"],
                "path": str(asset_path),
                "text": responses.get(file.name, ""),
            }
        )
    return sections


def build_manual_markdown(sections: List[dict]) -> str:
    """Create a Markdown manual containing the AI descriptions and saved image paths."""
    lines: List[str] = [
        "# 自動生成マニュアル",
        "",
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for idx, section in enumerate(sections, start=1):
        lines.append(f"## Step {idx}: {section['name']}")
        lines.append("")
        body = section.get("text") or "説明が入力されていません。"
        lines.append(body.strip())
        source_path = section.get("path")
        if source_path:
            lines.append("")
            lines.append(f"- 参照画像: `{source_path}`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_json_to_disk(payload: dict) -> str:
    """Persist JSON to a timestamped file and return the path."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = DATA_DIR / f"manual-{timestamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def main() -> None:
    _init_state()

    st.title("Create Manual From images")
    st.caption("複数画像の様子を読み取りテキスト化し、マニュアル生成を支援します。")

    saved_images: Dict[str, Dict[str, str]] = {}

    with st.sidebar:
        st.header("操作メニュー")
        uploaded_files = st.file_uploader(
            "画像をアップロード",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True,
        )
        saved_images = _ensure_saved_images(uploaded_files or [])
        api_key_default = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            value=api_key_default,
            type="password",
            help="キーが未入力の場合は環境変数 API_KEY / OPENAI_API_KEY を参照します。",
        )

        if st.button("画像からテキスト化", disabled=not uploaded_files):
            if not api_key:
                st.error("OpenAI API Key を入力してください。")
            elif not saved_images:
                st.warning("画像をアップロードしてください。")
            else:
                client = OpenAI(api_key=api_key)
                sorted_files = sorted(uploaded_files, key=lambda f: f.name.lower())
                for file in sorted_files:
                    try:
                        asset = saved_images.get(file.name)
                        if not asset:
                            st.warning(f"{file.name} の保存済みファイルが見つかりません。")
                            continue
                        asset_path = Path(asset["path"])
                        if not asset_path.exists():
                            st.warning(f"{asset_path} が存在しません。")
                            continue
                        image_bytes = asset_path.read_bytes()
                        result = describe_image(client, file.name, image_bytes)
                        st.session_state["ai_responses"][file.name] = result["text"]
                        st.session_state["raw_texts"][file.name] = result["raw_blocks"]
                    except Exception:
                        break

        if st.button("マニュアル化", disabled=not st.session_state["ai_responses"]):
            if not uploaded_files:
                st.warning("先に画像をアップロードしてください。")
            elif not saved_images:
                st.warning("画像の保存に失敗しています。再度アップロードしてください。")
            else:
                sorted_files = sorted(uploaded_files, key=lambda f: f.name.lower())
                sections = build_manual_sections(
                    sorted_files,
                    st.session_state["ai_responses"],
                    saved_images,
                )
                manual_json = {"generated_at": datetime.now().isoformat(), "steps": sections}
                st.session_state["manual_json"] = manual_json
                st.session_state["manual_markdown"] = build_manual_markdown(sections)
                st.session_state["saved_json_path"] = save_json_to_disk(manual_json)
                st.success("マニュアルを生成しました。下部でご確認ください。")

    sorted_uploads = sorted(uploaded_files, key=lambda f: f.name.lower()) if uploaded_files else []

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("アップロード画像")
        if not sorted_uploads:
            st.info("サイドバーから画像をアップロードしてください。")
        else:
            for file in sorted_uploads:
                image = file.read()
                st.image(image, caption=file.name, use_container_width=True)
                file.seek(0)

    with col_right:
        st.subheader("AIレスポンス")
        if not sorted_uploads:
            st.info("画像がアップロードされるとここに結果が表示されます。")
        else:
            for file in sorted_uploads:
                default_text = st.session_state["ai_responses"].get(file.name, "")
                st.markdown(f"**{file.name}**")
                st.write(default_text or "_結果が入力されていません。_")
                raw_blocks = st.session_state["raw_texts"].get(file.name)
                if raw_blocks:
                    st.caption(f"OpenAIレスポンス {file.name}")
                    st.code("\n\n".join(raw_blocks), language="markdown")

    st.subheader("生成されたマニュアル (Markdown)")
    if st.session_state["manual_markdown"]:
        st.download_button(
            "マニュアルをダウンロード (.md)",
            data=st.session_state["manual_markdown"],
            file_name="manual.md",
            mime="text/markdown",
        )

        if st.session_state["manual_json"]:
            st.download_button(
                "JSONをダウンロード",
                data=json.dumps(st.session_state["manual_json"], ensure_ascii=False, indent=2),
                file_name="manual.json",
                mime="application/json",
            )

        if st.session_state["saved_json_path"]:
            st.caption(f"JSONはローカルにも保存済み: `{st.session_state['saved_json_path']}`")
    else:
        st.info("「マニュアル化」ボタンを押すとここに表示されます。")


if __name__ == "__main__":
    main()
