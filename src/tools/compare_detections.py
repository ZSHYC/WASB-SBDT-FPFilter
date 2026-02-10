import os
import csv
import re
import math
import argparse
from pathlib import Path
from collections import defaultdict


DATASETS_DEFAULT = [
    "1",
    "2",
    "3",
    "land",
    # "maibo_land",
    # "maibo_serve_left",
    # "maibo_serve_right",
    "qingzhen"
]


def _extract_ints(s: str):
    nums = re.findall(r"(\d+)", s)
    return tuple(int(n) for n in nums) if nums else None


def _dataset_sort_key(ds: str):
    # If dataset name starts with a number, sort by that number first
    m = re.match(r"^(\d+)$", ds)
    if m:
        return (0, int(m.group(1)), ds)
    # otherwise put after numeric datasets and sort by name
    return (1, ds)


def _game_sort_key(game: str):
    ints = _extract_ints(game)
    if ints:
        return (0, ints, game)
    return (1, game)


def _image_sort_key(img: str):
    ints = _extract_ints(img)
    if ints:
        return (0, ints, img)
    return (1, img)



def find_wasb_detections(wasb_dataset_dir: Path):
    """
    Scan wasb outputs folder for the dataset. Returns dict: game -> {image_stem: detected}
    """
    results = defaultdict(dict)
    if not wasb_dataset_dir.exists():
        return results

    for f in wasb_dataset_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if not name.lower().endswith("predictions.csv"):
            continue

        # derive game name from filename (strip predictions suffix)
        raw_game = name[: name.lower().rfind("_predictions.csv")]
        # remove suffix like _Clip1 or _Clip_1 (case-insensitive), treat game_1_Clip1 as game_1
        game = re.sub(r'(?i)_clip[_-]?\d*$', '', raw_game)

        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            # heuristics to find image, x, y columns
            fieldnames = [fn.lower() for fn in reader.fieldnames or []]

            def find_field(patterns):
                for p in patterns:
                    for fn in reader.fieldnames or []:
                        if re.search(p, fn, re.IGNORECASE):
                            return fn
                return None

            img_col = find_field([r"(^|_)img", r"image", r"file", r"frame"])
            x_col = find_field([r"x[_ -]?coord", r"x[_ -]?coordinate", r"^x$", r"xpos"])
            y_col = find_field([r"y[_ -]?coord", r"y[_ -]?coordinate", r"^y$", r"ypos"])

            # fallback: inspect first row to detect numeric-like columns
            first_row = None
            try:
                first_row = next(reader)
            except StopIteration:
                first_row = None

            # if we consumed one row, we need to re-open to iterate all
            if first_row is not None:
                fh.seek(0)
                reader = csv.DictReader(fh)

            if not x_col or not y_col:
                # try to find columns that contain -inf in file or numeric
                candidates = []
                for fn in reader.fieldnames or []:
                    candidates.append(fn)
                # pick first two candidate numeric-like columns
                numeric_cols = []
                for fn in candidates:
                    # check a few rows to detect numeric pattern
                    fh.seek(0)
                    r = csv.DictReader(fh)
                    count_numeric = 0
                    checked = 0
                    for row in r:
                        v = (row.get(fn) or "").strip()
                        checked += 1
                        if v == "":
                            continue
                        if re.match(r"^-?inf$", v, re.IGNORECASE) or re.match(r"^-?\d+(\.\d+)?$", v):
                            count_numeric += 1
                        if checked >= 5:
                            break
                    if count_numeric >= 1:
                        numeric_cols.append(fn)
                    if len(numeric_cols) >= 2:
                        break

                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]

            # If still no img_col, attempt common names
            if not img_col:
                for fn in reader.fieldnames or []:
                    if re.search(r"img|image|file|frame", fn, re.IGNORECASE):
                        img_col = fn
                        break

            # iterate rows and decide detected or not
            for row in reader:
                img_val = None
                if img_col:
                    img_val = (row.get(img_col) or "").strip()
                else:
                    # fallback: try to find a filename-like column
                    for k, v in row.items():
                        if v and re.search(r"\.(jpg|png|jpeg)$", v, re.IGNORECASE):
                            img_val = v.strip()
                            break

                if not img_val:
                    # attempt to use row index as image id
                    # create a pseudo id using line number if no image found
                    img_val = f"row_{reader.line_num}"

                img_stem = Path(img_val).stem

                detected = False
                if x_col and y_col:
                    xv = (row.get(x_col) or "").strip()
                    yv = (row.get(y_col) or "").strip()
                    try:
                        xv_f = float(xv)
                        yv_f = float(yv)
                        if math.isfinite(xv_f) and math.isfinite(yv_f):
                            detected = True
                        else:
                            detected = False
                    except Exception:
                        # treat non-numeric or -inf as not detected
                        detected = False
                else:
                    # If x/y not found: try to infer from any numeric-like column
                    detected = False
                    for k, v in row.items():
                        vs = (v or "").strip()
                        if vs == "":
                            continue
                        if re.match(r"^-?inf$", vs, re.IGNORECASE):
                            continue
                        if re.match(r"^-?\d+(\.\d+)?$", vs):
                            detected = True
                            break

                results[game][img_stem] = detected

    return results


def find_yolo_detections(yolo_dataset_dir: Path):
    """
    Scan yolo output folder for the dataset. Returns dict: game -> {image_stem: detected}
    """
    results = defaultdict(dict)
    if not yolo_dataset_dir.exists():
        return results

    for game_dir in sorted(yolo_dataset_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        game = game_dir.name
        labels_dir = game_dir / "labels"
        if not labels_dir.exists() or not labels_dir.is_dir():
            continue

        for txt in labels_dir.glob("*.txt"):
            img_stem = txt.stem
            detected = False
            try:
                # empty file or only whitespace -> not detected
                with txt.open("r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        if line.strip():
                            detected = True
                            break
            except Exception:
                detected = False

            results[game][img_stem] = detected

    return results


def combine_and_write(out_csv: Path, datasets, wasb_root: Path, yolo_root: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "game",
        "image",
        "wasb_detected",
        "yolo_detected",
        "status",
    ]

    totals = defaultdict(lambda: defaultdict(int))

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        # sort datasets with numeric-first order
        for ds in sorted(datasets, key=_dataset_sort_key):
            wasb_map = find_wasb_detections(wasb_root / ds)
            yolo_map = find_yolo_detections(yolo_root / ds)

            # union of games
            games = set(list(wasb_map.keys()) + list(yolo_map.keys()))
            if not games:
                # still allow empty dataset but continue
                continue

            # sort games by numeric components when possible
            for game in sorted(games, key=_game_sort_key):
                imgs = set()
                imgs.update(wasb_map.get(game, {}).keys())
                imgs.update(yolo_map.get(game, {}).keys())

                for img in sorted(imgs, key=_image_sort_key):
                    wasb_det = wasb_map.get(game, {}).get(img, False)
                    yolo_det = yolo_map.get(game, {}).get(img, False)

                    if wasb_det and yolo_det:
                        status = "both"
                        totals[ds]["both"] += 1
                    elif wasb_det and not yolo_det:
                        status = "only_wasb"
                        totals[ds]["only_wasb"] += 1
                    elif (not wasb_det) and yolo_det:
                        status = "only_yolo"
                        totals[ds]["only_yolo"] += 1
                    else:
                        status = "neither"
                        totals[ds]["neither"] += 1

                    writer.writerow(
                        {
                            "dataset": ds,
                            "game": game,
                            "image": img,
                            "wasb_detected": str(wasb_det),
                            "yolo_detected": str(yolo_det),
                            "status": status,
                        }
                    )

    # Also write a simple summary next to CSV
    summary_path = out_csv.with_name(out_csv.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as sf:
        s_writer = csv.writer(sf)
        s_writer.writerow(["dataset", "both", "only_wasb", "only_yolo", "neither"])
        for ds in sorted(datasets, key=_dataset_sort_key):
            s = totals.get(ds, {})
            s_writer.writerow([
                ds,
                s.get("both", 0),
                s.get("only_wasb", 0),
                s.get("only_yolo", 0),
                s.get("neither", 0),
            ])


def parse_args():
    p = argparse.ArgumentParser(description="Compare WASB vs YOLO detections and output CSV report")
    p.add_argument(
        "--wasb-root",
        default=Path("wasb_outputs"),
        type=Path,
        help="Root folder for wasb outputs (default: wasb_outputs)",
    )
    p.add_argument(
        "--yolo-root",
        default=Path("yolo_output"),
        type=Path,
        help="Root folder for yolo outputs (default: yolo_output)",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=DATASETS_DEFAULT,
        help="List of datasets to check (default: the 7 requested)",
    )
    p.add_argument(
        "--out-csv",
        default=Path("reports/detection_comparison.csv"),
        type=Path,
        help="Output CSV path",
    )
    return p.parse_args()


def main():
    args = parse_args()

    wasb_root = Path(args.wasb_root)
    yolo_root = Path(args.yolo_root)
    datasets = args.datasets

    print(f"WASB root: {wasb_root}")
    print(f"YOLO root: {yolo_root}")
    print(f"Datasets: {datasets}")
    print(f"Writing output to: {args.out_csv}")

    combine_and_write(args.out_csv, datasets, wasb_root, yolo_root)

    print("Done. CSV and summary written.")


if __name__ == "__main__":
    main()


# python d:\Personal\Desktop\WASB-SBDT-FPFilter\src\tools\compare_detections.py --wasb-root "D:\Personal\Desktop\WASB-SBDT-FPFilter\wasb_outputs" --yolo-root "D:\Personal\Desktop\yolo\yolo_output" --out-csv "D:\Personal\Desktop\WASB-SBDT-FPFilter\reports\detection_comparison.csv"