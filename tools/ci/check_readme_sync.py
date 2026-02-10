#!/usr/bin/env python3
"""Simple CI check to ensure README_en.md and README_zh.md remain structurally synchronized.

Checks performed:
- Both files exist
- The sequence of heading levels (ignoring top-level H1) is identical in both files
- Both files contain the arXiv paper link (arxiv.org/abs/2311.05237)
- Both files mention 'run_inference_pipeline.py'

Exit codes:
- 0: pass
- 1: fail (messages printed to stdout)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EN = ROOT / 'README_en.md'
ZH = ROOT / 'README_zh.md'

FAIL = False
msgs = []


def extract_heading_levels(path):
    levels = []
    headings = []
    for ln in path.read_text(encoding='utf-8').splitlines():
        s = ln.strip()
        if s.startswith('#'):
            # count # at start
            i = 0
            while i < len(s) and s[i] == '#':
                i += 1
            # store level and raw text
            text = s[i:].strip()
            levels.append(i)
            headings.append(text)
    return levels, headings


if not EN.exists():
    print(f"ERROR: {EN} not found")
    sys.exit(1)
if not ZH.exists():
    print(f"ERROR: {ZH} not found")
    sys.exit(1)

levels_en, heads_en = extract_heading_levels(EN)
levels_zh, heads_zh = extract_heading_levels(ZH)

# ignore first H1 (top title) if present
if levels_en and levels_en[0] == 1:
    levels_en = levels_en[1:]
    heads_en = heads_en[1:]
if levels_zh and levels_zh[0] == 1:
    levels_zh = levels_zh[1:]
    heads_zh = heads_zh[1:]

if len(levels_en) != len(levels_zh):
    FAIL = True
    msgs.append(f"Heading count mismatch: README_en has {len(levels_en)} headings (excluding H1), README_zh has {len(levels_zh)} headings (excluding H1)")
else:
    diffs = []
    for idx, (le, lz) in enumerate(zip(levels_en, levels_zh)):
        if le != lz:
            diffs.append((idx + 1, le, lz, heads_en[idx] if idx < len(heads_en) else '', heads_zh[idx] if idx < len(heads_zh) else ''))
    if diffs:
        FAIL = True
        msgs.append("Heading level mismatches (position, en_level, zh_level, en_heading, zh_heading):")
        for item in diffs:
            msgs.append(str(item))

# check paper link present
paper_link = 'arxiv.org/abs/2311.05237'
for path in (EN, ZH):
    txt = path.read_text(encoding='utf-8')
    if paper_link not in txt:
        FAIL = True
        msgs.append(f"Missing paper link '{paper_link}' in {path.name}")

# check pipeline mention present
for path in (EN, ZH):
    txt = path.read_text(encoding='utf-8')
    if 'run_inference_pipeline.py' not in txt:
        FAIL = True
        msgs.append(f"Missing mention of 'run_inference_pipeline.py' in {path.name}")

if FAIL:
    print('\n'.join(msgs))
    sys.exit(1)
else:
    print('README sync check passed ✅')
    sys.exit(0)
