import argparse
import csv
import re
from pathlib import Path


AVG_RE = re.compile(r"Average Accuracy:\s*([0-9.]+)")
LAST_RE = re.compile(r"Last Accuracy:\s*([0-9.]+)")
CURVE_RE = re.compile(r"CNN top1 curve:\s*\[([^\]]*)\]")
FORGET_RE = re.compile(r"Forgetting:\s*([-0-9.]+)\s*Backward:\s*([-0-9.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    avg = [float(x) for x in AVG_RE.findall(text)]
    last = [float(x) for x in LAST_RE.findall(text)]
    curves = CURVE_RE.findall(text)
    forget = FORGET_RE.findall(text)

    curve = ""
    if curves:
        curve = ",".join(x.strip() for x in curves[-1].split(",") if x.strip())

    return {
        "log": str(path),
        "average_accuracy": avg[-1] if avg else "",
        "last_accuracy": last[-1] if last else "",
        "top1_curve": curve,
        "final_forgetting": forget[-1][0] if forget else "",
        "final_backward": forget[-1][1] if forget else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse featmatch CIL logs into a CSV evidence table.")
    parser.add_argument("logdir", help="Directory containing .log files.")
    parser.add_argument("--out", default="results_summary.csv", help="Output CSV path.")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    rows = [parse_log(path) for path in sorted(logdir.rglob("*.log"))]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "log",
                "average_accuracy",
                "last_accuracy",
                "top1_curve",
                "final_forgetting",
                "final_backward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

