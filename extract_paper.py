#!/usr/bin/env python3
"""
extract_paper.py — Extract figures, equations, and metrics from a research paper PDF.

General-purpose tool for Phase 0 of paper implementation.

Usage:
    uv run extract_paper.py paper/paper.pdf
    uv run extract_paper.py paper/paper.pdf --output-dir paper/

Outputs:
    paper/images/fig_<N>.png    — Extracted figure images
    paper/images/fig_<N>.txt    — Annotations (caption, in-figure text, body references)
    paper/equations.md          — All numbered equations with context
    paper/metrics.md            — All reported metrics from tables
"""

import argparse
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def get_text_blocks(page):
    """Extract text blocks with bounding boxes from a page."""
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks = []
    for b in raw["blocks"]:
        if b["type"] != 0:  # skip image blocks
            continue
        lines_text = []
        for line in b["lines"]:
            spans_text = "".join(s["text"] for s in line["spans"])
            lines_text.append(spans_text)
        text = "\n".join(lines_text)
        if text.strip():
            blocks.append({"text": text, "bbox": fitz.Rect(b["bbox"])})
    return blocks


def get_full_text(doc):
    """Get the full text of the document, page by page."""
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return pages


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------

FIG_CAPTION_RE = re.compile(
    r'((?:Figure|Fig\.?)\s+(\d+)[.:\s].*?)(?=\n\n|\n(?:Figure|Fig\.?)\s+\d|\Z)',
    re.IGNORECASE | re.DOTALL,
)


def find_figure_captions(doc):
    """Find all figure captions with page number and bounding box."""
    figures = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = get_text_blocks(page)
        for block in blocks:
            for m in FIG_CAPTION_RE.finditer(block["text"]):
                fig_num = int(m.group(2))
                if fig_num not in figures:
                    figures[fig_num] = {
                        "num": fig_num,
                        "page": page_num,
                        "caption": m.group(1).strip(),
                        "bbox": block["bbox"],
                    }
    return sorted(figures.values(), key=lambda f: f["num"])


def extract_figure_image(doc, fig_info, dpi=200):
    """Extract a figure image, trying embedded images first, then page crop."""
    page = doc[fig_info["page"]]
    caption_bbox = fig_info["bbox"]

    # --- Strategy 1: find a large embedded image near the caption ---
    images = page.get_images(full=True)
    best_img = None
    best_dist = float("inf")

    for img_ref in images:
        xref = img_ref[0]
        width, height = img_ref[2], img_ref[3]
        if width < 100 or height < 100:
            continue

        # Find where this image is placed on the page
        img_rects = page.get_image_rects(xref)
        for rect in img_rects:
            # Distance from image bottom to caption top (figure is above caption)
            dist = abs(rect.y1 - caption_bbox.y0)
            if dist < best_dist and rect.y1 <= caption_bbox.y1:
                best_dist = dist
                best_img = xref

    if best_img is not None and best_dist < 200:
        img_data = doc.extract_image(best_img)
        return img_data["image"], img_data["ext"]

    # --- Strategy 2: crop from rendered page ---
    page_rect = page.rect

    # Figure region: full width, from estimated top to caption bottom
    # Heuristic: look for the nearest block boundary above the caption
    blocks = get_text_blocks(page)
    blocks_above = [
        b for b in blocks
        if b["bbox"].y1 < caption_bbox.y0 - 5
        and not FIG_CAPTION_RE.search(b["text"])
    ]

    if blocks_above:
        # Figure starts just below the last body-text block above caption
        nearest_above = max(blocks_above, key=lambda b: b["bbox"].y1)
        fig_top = nearest_above["bbox"].y1 + 2
    else:
        fig_top = max(0, caption_bbox.y0 - page_rect.height * 0.45)

    fig_bottom = caption_bbox.y1 + 5
    margin = 15
    clip = fitz.Rect(margin, fig_top, page_rect.width - margin, fig_bottom)

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    return pix.tobytes("png"), "png"


def get_figure_in_text(doc, fig_info, dpi=150):
    """Extract text that appears within the figure region on the page."""
    page = doc[fig_info["page"]]
    caption_bbox = fig_info["bbox"]
    page_rect = page.rect

    # Approximate figure region (above caption)
    fig_top = max(0, caption_bbox.y0 - page_rect.height * 0.4)
    fig_rect = fitz.Rect(0, fig_top, page_rect.width, caption_bbox.y0)

    blocks = get_text_blocks(page)
    in_fig_texts = []
    for b in blocks:
        if fig_rect.intersects(b["bbox"]) and not FIG_CAPTION_RE.search(b["text"]):
            text = b["text"].strip()
            if len(text) > 2:
                in_fig_texts.append(text)
    return in_fig_texts


def find_figure_references(full_pages_text, fig_num):
    """Find all sentences in the paper body that reference a figure."""
    refs = []
    fig_patterns = [
        rf'(?:Figure|Fig\.?)\s+{fig_num}\b',
    ]
    combined = re.compile("|".join(fig_patterns), re.IGNORECASE)

    for page_num, page_text in enumerate(full_pages_text):
        sentences = re.split(r'(?<=[.!?])\s+', page_text)
        for sent in sentences:
            if combined.search(sent):
                sent_clean = " ".join(sent.split())
                if len(sent_clean) > 10:
                    refs.append(f"[p{page_num + 1}] {sent_clean}")
    return refs


def extract_figures(doc, output_dir):
    """Extract all figures with images and annotations."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    captions = find_figure_captions(doc)
    full_text = get_full_text(doc)

    print(f"Found {len(captions)} figures")

    for fig in captions:
        fig_num = fig["num"]

        # Extract image
        try:
            img_bytes, ext = extract_figure_image(doc, fig)
            img_path = images_dir / f"fig_{fig_num}.png"
            if ext == "png":
                img_path.write_bytes(img_bytes)
            else:
                # Convert to PNG via PyMuPDF
                pix = fitz.Pixmap(img_bytes)
                pix.save(str(img_path))
            print(f"  fig_{fig_num}.png — extracted from page {fig['page'] + 1}")
        except Exception as e:
            print(f"  fig_{fig_num}.png — FAILED: {e}")
            continue

        # Build annotation
        annotation_parts = []
        annotation_parts.append(f"Caption: {fig['caption']}")
        annotation_parts.append("")

        # In-figure text
        in_fig = get_figure_in_text(doc, fig)
        if in_fig:
            annotation_parts.append("In-figure text:")
            for t in in_fig:
                annotation_parts.append(f"  {t}")
            annotation_parts.append("")

        # References from paper body
        refs = find_figure_references(full_text, fig_num)
        if refs:
            annotation_parts.append("Referenced in paper:")
            for r in refs:
                annotation_parts.append(f"  {r}")
            annotation_parts.append("")

        txt_path = images_dir / f"fig_{fig_num}.txt"
        txt_path.write_text("\n".join(annotation_parts), encoding="utf-8")

    return captions


# ---------------------------------------------------------------------------
# Equation extraction
# ---------------------------------------------------------------------------

# Matches "(1)", "(2)", "(A.1)", etc. typically at end of a text block
EQ_NUM_RE = re.compile(r'\((\d+(?:\.\d+)?|[A-Z]\.\d+)\)\s*$', re.MULTILINE)


def find_equations(doc):
    """Find numbered equations in the paper."""
    equations = []
    seen = set()
    full_text = get_full_text(doc)

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = get_text_blocks(page)

        for block in blocks:
            for m in EQ_NUM_RE.finditer(block["text"]):
                eq_num = m.group(1)
                if eq_num in seen:
                    continue
                seen.add(eq_num)

                # The equation text is the block containing the number
                eq_text = block["text"].strip()

                # Get surrounding context from the page
                context_lines = []
                page_text = full_text[page_num]
                # Find sentences near the equation number
                pattern = re.compile(
                    rf'(?:[^.]*\(\s*{re.escape(eq_num)}\s*\)[^.]*\.)',
                    re.DOTALL,
                )
                for cm in pattern.finditer(page_text):
                    ctx = " ".join(cm.group(0).split())
                    if len(ctx) > 10:
                        context_lines.append(ctx)

                # Also find references to "Eq. N" or "Equation N" in the paper
                eq_refs = []
                eq_ref_re = re.compile(
                    rf'(?:Eq\.?|Equation)\s*[\(]?\s*{re.escape(eq_num)}\s*[\)]?',
                    re.IGNORECASE,
                )
                for pn, pt in enumerate(full_text):
                    sents = re.split(r'(?<=[.!?])\s+', pt)
                    for s in sents:
                        if eq_ref_re.search(s) and len(s.strip()) > 10:
                            eq_refs.append(f"[p{pn + 1}] {' '.join(s.split())}")

                equations.append({
                    "num": eq_num,
                    "text": eq_text,
                    "context": context_lines,
                    "references": eq_refs,
                    "page": page_num,
                })

    return sorted(equations, key=lambda e: (
        # Sort: plain numbers first, then lettered
        0 if e["num"].isdigit() else 1,
        e["num"],
    ))


def write_equations(equations, output_dir):
    """Write equations.md."""
    lines = ["# Equations", "", f"Extracted {len(equations)} numbered equations.", ""]

    for eq in equations:
        lines.append(f"## Equation ({eq['num']})  —  page {eq['page'] + 1}")
        lines.append("")
        lines.append("```")
        lines.append(eq["text"])
        lines.append("```")
        lines.append("")

        if eq["context"]:
            lines.append("**Surrounding context:**")
            for c in eq["context"]:
                lines.append(f"> {c}")
            lines.append("")

        if eq["references"]:
            lines.append("**Referenced in:**")
            for r in eq["references"]:
                lines.append(f"- {r}")
            lines.append("")

        lines.append("---")
        lines.append("")

    (output_dir / "equations.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote equations.md ({len(equations)} equations)")


# ---------------------------------------------------------------------------
# Metrics / table extraction
# ---------------------------------------------------------------------------

TABLE_CAPTION_RE = re.compile(
    r'(Table\s+(\d+)[.:\s].*?)(?=\n\n|\n(?:Table)\s+\d|\Z)',
    re.IGNORECASE | re.DOTALL,
)


def find_tables(doc):
    """Find tables and extract their content."""
    tables = {}
    full_text = get_full_text(doc)

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = get_text_blocks(page)

        for block in blocks:
            for m in TABLE_CAPTION_RE.finditer(block["text"]):
                tbl_num = int(m.group(2))
                if tbl_num not in tables:
                    tables[tbl_num] = {
                        "num": tbl_num,
                        "page": page_num,
                        "caption": m.group(1).strip(),
                        "bbox": block["bbox"],
                    }

    # For each table, extract text below the caption on the same page
    result = []
    for tbl in sorted(tables.values(), key=lambda t: t["num"]):
        page = doc[tbl["page"]]
        page_rect = page.rect
        blocks = get_text_blocks(page)

        # Table content is typically below the caption
        tbl_blocks = [
            b for b in blocks
            if b["bbox"].y0 >= tbl["bbox"].y1 - 5
            and b["bbox"].y0 < tbl["bbox"].y1 + page_rect.height * 0.5
            and not TABLE_CAPTION_RE.search(b["text"])
            and not FIG_CAPTION_RE.search(b["text"])
        ]
        tbl_blocks.sort(key=lambda b: (b["bbox"].y0, b["bbox"].x0))

        content_lines = []
        for b in tbl_blocks:
            text = b["text"].strip()
            if text:
                content_lines.append(text)
            # Stop if we hit a paragraph of body text (heuristic: >100 chars, no numbers)
            if len(text) > 100 and not re.search(r'\d+\.\d+', text):
                break

        # Find references to this table
        refs = []
        tbl_ref_re = re.compile(rf'Table\s+{tbl["num"]}\b', re.IGNORECASE)
        for pn, pt in enumerate(full_text):
            sents = re.split(r'(?<=[.!?])\s+', pt)
            for s in sents:
                if tbl_ref_re.search(s) and len(s.strip()) > 10:
                    clean = " ".join(s.split())
                    if clean not in [r.split("] ", 1)[-1] if "] " in r else r for r in refs]:
                        refs.append(f"[p{pn + 1}] {clean}")

        result.append({
            "num": tbl["num"],
            "page": tbl["page"],
            "caption": tbl["caption"],
            "content": "\n".join(content_lines),
            "references": refs,
        })

    return result


def write_metrics(tables, output_dir):
    """Write metrics.md."""
    lines = ["# Reported Metrics", "", f"Extracted {len(tables)} tables.", ""]

    for tbl in tables:
        lines.append(f"## Table {tbl['num']}  —  page {tbl['page'] + 1}")
        lines.append("")
        lines.append(f"**Caption:** {tbl['caption']}")
        lines.append("")
        if tbl["content"]:
            lines.append("**Content:**")
            lines.append("```")
            lines.append(tbl["content"])
            lines.append("```")
            lines.append("")
        if tbl["references"]:
            lines.append("**Referenced in:**")
            for r in tbl["references"]:
                lines.append(f"- {r}")
            lines.append("")
        lines.append("---")
        lines.append("")

    (output_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote metrics.md ({len(tables)} tables)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract figures, equations, and metrics from a paper PDF.",
    )
    parser.add_argument("pdf", help="Path to the paper PDF")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same directory as the PDF)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for figure rendering")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting from: {pdf_path}")
    print(f"Output to: {output_dir}")
    print()

    doc = fitz.open(str(pdf_path))
    print(f"PDF: {len(doc)} pages")
    print()

    # Extract all three types
    print("--- Figures ---")
    extract_figures(doc, output_dir)
    print()

    print("--- Equations ---")
    equations = find_equations(doc)
    write_equations(equations, output_dir)
    print()

    print("--- Tables / Metrics ---")
    tables = find_tables(doc)
    write_metrics(tables, output_dir)
    print()

    doc.close()
    print("Done.")


if __name__ == "__main__":
    main()
