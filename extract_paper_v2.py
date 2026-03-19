#!/usr/bin/env python3
"""
extract_paper_v2.py — Improved extraction for LaTeX-generated PDFs.

Handles:
- Vector graphics figures (rendered from page)
- Multi-column layouts
- Equations from display math
- Table content

Usage:
    python extract_paper_v2.py paper/CarPlanner.pdf --output-dir paper/
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import fitz


@dataclass
class FigureInfo:
    num: int
    caption: str
    caption_page: int
    caption_bbox: fitz.Rect
    figure_page: int
    figure_bbox: Optional[fitz.Rect] = None


def get_text_blocks(page):
    """Extract text blocks with bounding boxes."""
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks = []
    for b in raw["blocks"]:
        if b["type"] != 0:
            continue
        lines_text = []
        for line in b["lines"]:
            spans_text = "".join(s["text"] for s in line["spans"])
            lines_text.append(spans_text)
        text = "\n".join(lines_text)
        if text.strip():
            blocks.append({"text": text, "bbox": fitz.Rect(b["bbox"])})
    return blocks


def find_figure_captions_improved(doc):
    """Find figure captions with improved multi-page handling.

    Rules:
    - Skip page 1 (title page)
    - Prefer longer, more complete captions over short fragments
    - Look for captions in proper figure regions (near drawings)
    """
    figures = {}

    for page_num in range(1, len(doc)):  # Skip page 0 (title page)
        page = doc[page_num]
        text = page.get_text()
        blocks = get_text_blocks(page)
        drawings = list(page.get_drawings())

        for block in blocks:
            block_text = block["text"]

            for m in re.finditer(
                r"(?:Figure|Fig\.)\s*(\d+)[.:\s]+([^\n]+(?:\n(?![A-Z][a-z])[\w\s]+)*)",
                block_text,
                re.IGNORECASE | re.MULTILINE,
            ):
                fig_num = int(m.group(1))
                caption_text = m.group(0).strip()

                # Prefer captions that are more than just "Figure N."
                caption_content = (
                    caption_text.split(".", 1)[-1].strip()
                    if "." in caption_text
                    else ""
                )
                has_real_content = len(caption_content) > 10

                # Check if this page has drawings (likely a figure page)
                has_drawings = len(drawings) > 10

                existing = figures.get(fig_num)
                should_replace = (
                    existing is None
                    or (
                        has_drawings and not figures[fig_num].get("has_drawings", False)
                    )
                    or (
                        has_real_content
                        and len(caption_text) > len(existing.get("caption", ""))
                    )
                )

                if should_replace:
                    figures[fig_num] = {
                        "num": fig_num,
                        "caption": caption_text,
                        "caption_page": page_num,
                        "caption_bbox": block["bbox"],
                        "has_drawings": has_drawings,
                    }

    return figures


def find_figure_regions(doc, figures):
    """Find the actual figure regions using drawing density analysis."""
    for fig_num, fig_info in figures.items():
        caption_page = fig_info["caption_page"]
        caption_bbox = fig_info["caption_bbox"]

        page = doc[caption_page]
        page_rect = page.rect

        drawings = list(page.get_drawings())

        if drawings:
            y_positions = []
            for d in drawings:
                if d.get("rect"):
                    y_positions.append((d["rect"].y0, d["rect"].y1))

            if y_positions:
                drawing_top = min(y for y, _ in y_positions)
                drawing_bottom = max(y for _, y in y_positions)

                caption_y = caption_bbox.y0

                if drawing_bottom < caption_y:
                    fig_top = max(0, drawing_top - 20)
                    fig_bottom = min(caption_y - 5, drawing_bottom + 20)
                else:
                    blocks = get_text_blocks(page)
                    blocks_above = [
                        b
                        for b in blocks
                        if b["bbox"].y1 < caption_y - 10
                        and not re.search(
                            r"(?:Figure|Fig\.)\s*\d+", b["text"], re.IGNORECASE
                        )
                    ]

                    if blocks_above:
                        nearest = max(blocks_above, key=lambda b: b["bbox"].y1)
                        fig_top = nearest["bbox"].y1 + 5
                    else:
                        fig_top = max(0, caption_y - page_rect.height * 0.4)

                    fig_bottom = caption_y - 5

                fig_info["figure_page"] = caption_page
                fig_info["figure_bbox"] = fitz.Rect(
                    10, fig_top, page_rect.width - 10, fig_bottom
                )
        else:
            fig_info["figure_page"] = caption_page
            fig_info["figure_bbox"] = fitz.Rect(
                10,
                max(0, caption_bbox.y0 - page_rect.height * 0.35),
                page_rect.width - 10,
                caption_bbox.y0 - 5,
            )

    return figures


def extract_figure_image(doc, fig_info, dpi=200):
    """Render figure region from page."""
    page = doc[fig_info["figure_page"]]
    fig_bbox = fig_info.get("figure_bbox")

    if fig_bbox is None:
        page_rect = page.rect
        fig_bbox = fitz.Rect(
            10, 0, page_rect.width - 10, fig_info["caption_bbox"].y0 - 5
        )

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=fig_bbox)
    return pix.tobytes("png"), "png"


def get_figure_in_text(doc, fig_info):
    """Extract text labels within the figure region."""
    page = doc[fig_info["figure_page"]]
    fig_bbox = fig_info.get("figure_bbox")

    if fig_bbox is None:
        return []

    blocks = get_text_blocks(page)
    in_fig_texts = []

    for b in blocks:
        if fig_bbox.intersects(b["bbox"]):
            text = b["text"].strip()
            if len(text) > 2 and not re.search(
                r"(?:Figure|Fig\.)\s*\d+", text, re.IGNORECASE
            ):
                in_fig_texts.append(text)

    return in_fig_texts


def find_figure_references(doc, fig_num):
    """Find all references to this figure in the paper."""
    refs = []
    patterns = [
        rf"(?:Figure|Fig\.?)\s+{fig_num}\b",
        rf"(?:Figure|Fig\.?)\s+{fig_num}[a-z]?\b",
    ]
    combined = re.compile("|".join(patterns), re.IGNORECASE)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sent in sentences:
            if combined.search(sent):
                sent_clean = " ".join(sent.split())
                if len(sent_clean) > 15 and f"Figure {fig_num}" not in sent_clean[:20]:
                    refs.append(f"[p{page_num + 1}] {sent_clean[:200]}")

    return refs[:10]


def extract_figures(doc, output_dir):
    """Extract all figures with improved handling."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    figures = find_figure_captions_improved(doc)
    figures = find_figure_regions(doc, figures)

    print(f"Found {len(figures)} figures")

    for fig_num in sorted(figures.keys()):
        fig_info = figures[fig_num]

        try:
            img_bytes, ext = extract_figure_image(doc, fig_info)
            img_path = images_dir / f"fig_{fig_num}.png"
            img_path.write_bytes(img_bytes)
            print(f"  fig_{fig_num}.png — page {fig_info['figure_page'] + 1}")
        except Exception as e:
            print(f"  fig_{fig_num}.png — FAILED: {e}")
            continue

        annotation_parts = []
        annotation_parts.append(f"Caption: {fig_info['caption']}")
        annotation_parts.append(f"Location: Page {fig_info['figure_page'] + 1}")
        annotation_parts.append("")

        in_fig = get_figure_in_text(doc, fig_info)
        if in_fig:
            annotation_parts.append("In-figure text/labels:")
            for t in in_fig[:15]:
                annotation_parts.append(f"  {t[:100]}")
            annotation_parts.append("")

        refs = find_figure_references(doc, fig_num)
        if refs:
            annotation_parts.append("Referenced in paper:")
            for r in refs:
                annotation_parts.append(f"  {r}")
            annotation_parts.append("")

        txt_path = images_dir / f"fig_{fig_num}.txt"
        txt_path.write_text("\n".join(annotation_parts), encoding="utf-8")

    return figures


def find_equations_improved(doc):
    """Find equations using text analysis for display math patterns."""
    equations = []
    seen_nums = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        blocks = get_text_blocks(page)

        for block in blocks:
            block_text = block["text"]

            for m in re.finditer(r"\((\d+(?:\.\d+)?)\)\s*$", block_text, re.MULTILINE):
                eq_num = m.group(1)

                if eq_num in seen_nums:
                    continue
                seen_nums.add(eq_num)

                eq_text = block_text.strip()

                eq_refs = []
                ref_patterns = [
                    rf"Eq\.?\s*\(?\s*{re.escape(eq_num)}\s*\)?",
                    rf"Equation\s*\(?\s*{re.escape(eq_num)}\s*\)?",
                ]
                for ref_pat in ref_patterns:
                    for pn in range(len(doc)):
                        pt = doc[pn].get_text()
                        for ref_m in re.finditer(ref_pat, pt, re.IGNORECASE):
                            start = max(0, ref_m.start() - 100)
                            end = min(len(pt), ref_m.end() + 100)
                            ctx = pt[start:end].strip()
                            ctx = re.sub(r"\s+", " ", ctx)
                            if len(ctx) > 20:
                                eq_refs.append(f"[p{pn + 1}] ...{ctx}...")
                            break

                equations.append(
                    {
                        "num": eq_num,
                        "text": eq_text,
                        "page": page_num,
                        "references": eq_refs[:5],
                    }
                )

    return sorted(
        equations,
        key=lambda e: (
            0 if e["num"].replace(".", "").isdigit() else 1,
            float(e["num"]) if e["num"].replace(".", "").isdigit() else 0,
        ),
    )


def write_equations(equations, output_dir):
    """Write equations.md with cleaner formatting."""
    lines = ["# Equations\n", f"Extracted {len(equations)} numbered equations.\n"]

    for eq in equations:
        lines.append(f"## Equation ({eq['num']})  —  page {eq['page'] + 1}\n")
        lines.append("```")
        lines.append(eq["text"])
        lines.append("```\n")

        if eq["references"]:
            lines.append("**Referenced in:**")
            for r in eq["references"]:
                lines.append(f"- {r}")
            lines.append("")

        lines.append("---\n")

    (output_dir / "equations.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote equations.md ({len(equations)} equations)")


def find_tables_improved(doc):
    """Find tables with improved content extraction."""
    tables = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        blocks = get_text_blocks(page)

        for block in blocks:
            for m in re.finditer(
                r"(Table\s+(\d+)[.:\s]+[^\n]+)", block["text"], re.IGNORECASE
            ):
                tbl_num = int(m.group(2))
                caption = m.group(1).strip()

                if tbl_num not in tables:
                    tables[tbl_num] = {
                        "num": tbl_num,
                        "page": page_num,
                        "caption": caption,
                        "caption_bbox": block["bbox"],
                    }

    result = []
    for tbl_num in sorted(tables.keys()):
        tbl = tables[tbl_num]
        page = doc[tbl["page"]]
        blocks = get_text_blocks(page)

        table_blocks = []
        caption_y = tbl["caption_bbox"].y1

        for b in blocks:
            by = b["bbox"].y0
            if by >= caption_y - 5 and by < caption_y + 400:
                text = b["text"].strip()
                if text and not re.search(
                    r"(?:Table|Figure)\s+\d+", text, re.IGNORECASE
                ):
                    has_numbers = bool(re.search(r"\d+\.\d+|\d{2,}", text))
                    is_short = len(text) < 300
                    if has_numbers or is_short:
                        table_blocks.append(text)

        refs = []
        for pn in range(len(doc)):
            pt = doc[pn].get_text()
            for m in re.finditer(rf"Table\s+{tbl_num}\b", pt, re.IGNORECASE):
                start = max(0, m.start() - 50)
                end = min(len(pt), m.end() + 100)
                ctx = pt[start:end].strip()
                ctx = re.sub(r"\s+", " ", ctx)
                if len(ctx) > 20 and ctx not in [
                    r.split("] ")[-1] if "] " in r else r for r in refs
                ]:
                    refs.append(f"[p{pn + 1}] {ctx}")
                    break

        result.append(
            {
                "num": tbl_num,
                "page": tbl["page"],
                "caption": tbl["caption"],
                "content": "\n".join(table_blocks[:20]),
                "references": refs[:5],
            }
        )

    return result


def write_metrics(tables, output_dir):
    """Write metrics.md with table data."""
    lines = ["# Reported Metrics\n", f"Extracted {len(tables)} tables.\n"]

    for tbl in tables:
        lines.append(f"## Table {tbl['num']}  —  page {tbl['page'] + 1}\n")
        lines.append(f"**Caption:** {tbl['caption']}\n")

        if tbl["content"]:
            lines.append("**Content:**")
            lines.append("```")
            lines.append(tbl["content"])
            lines.append("```\n")

        if tbl["references"]:
            lines.append("**Referenced in:**")
            for r in tbl["references"]:
                lines.append(f"- {r}")
            lines.append("")

        lines.append("---\n")

    (output_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote metrics.md ({len(tables)} tables)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract figures, equations, and metrics from a paper PDF."
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

    print("--- Figures ---")
    extract_figures(doc, output_dir)
    print()

    print("--- Equations ---")
    equations = find_equations_improved(doc)
    write_equations(equations, output_dir)
    print()

    print("--- Tables / Metrics ---")
    tables = find_tables_improved(doc)
    write_metrics(tables, output_dir)
    print()

    doc.close()
    print("Done.")


if __name__ == "__main__":
    main()
