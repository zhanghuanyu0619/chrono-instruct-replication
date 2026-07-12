#!/usr/bin/env python3
"""Render + upload Hugging Face model cards for the fine-tuned ChronoGPT-Instruct
vintages, and flip each repo from private to public.

For each vintage tau in {1999, 2005, 2010, 2015, 2020, 2024} this script:
  1. Renders `scripts/model_card_template.md`, substituting {{VINTAGE}},
     {{CUTOFF_DATE}}, and {{BASE_MODEL}} for that vintage.
  2. Uploads the rendered text as `README.md` to
     `HZ0619/chrono-instruct-v1-{tau}1231` (HfApi.upload_file).
  3. Sets the repo public via HfApi.update_repo_settings(private=False).

Authentication: the token is read from the standard Hugging Face auth (a cached
`hf auth login` / `huggingface-cli login`, or the HF_TOKEN env var). No token is
hardcoded. You need a WRITE token whose namespace is HZ0619.

Usage
-----
    # dry run: render cards to out/model_cards/{tau}.md, upload nothing
    python scripts/push_model_cards.py --dry-run

    # upload cards AND make every repo public (the full deliverable)
    python scripts/push_model_cards.py

    # upload cards but leave visibility unchanged
    python scripts/push_model_cards.py --no-publish

    # only a subset of vintages
    python scripts/push_model_cards.py 2020 2024

Requires: huggingface_hub (already a dependency of this package, >=0.23).
"""
import argparse
import sys
from pathlib import Path

ALL_VINTAGES = [1999, 2005, 2010, 2015, 2020, 2024]

REPO_TEMPLATE = "HZ0619/chrono-instruct-v1-{vintage}1231"
BASE_TEMPLATE = "manelalab/chrono-gpt-v1-{vintage}1231"

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "scripts" / "model_card_template.md"
OUT_DIR = REPO_ROOT / "out" / "model_cards"


def render_card(template: str, vintage: int) -> str:
    """Substitute the per-vintage placeholders into the template."""
    return (
        template.replace("{{VINTAGE}}", str(vintage))
        .replace("{{CUTOFF_DATE}}", f"{vintage}-12-31")
        .replace("{{BASE_MODEL}}", BASE_TEMPLATE.format(vintage=vintage))
    )


def make_public(api, repo_id: str) -> None:
    """Flip a repo to public using the current huggingface_hub API.

    `update_repo_settings(private=False)` is the current method (huggingface_hub
    >= ~0.25). `update_repo_visibility` is the older, now-deprecated predecessor;
    we fall back to it so this also works on older hub installs.
    """
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(repo_id=repo_id, private=False)
    else:  # pragma: no cover - only on old huggingface_hub
        api.update_repo_visibility(repo_id=repo_id, private=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render + upload model cards and publish the ChronoGPT-Instruct repos.",
    )
    parser.add_argument(
        "vintages",
        nargs="*",
        type=int,
        help="Optional subset of vintages (default: all six).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render cards to out/model_cards/{tau}.md without uploading or changing visibility.",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Upload the card but do NOT change repo visibility (leave it private).",
    )
    args = parser.parse_args()

    vintages = args.vintages or ALL_VINTAGES
    unknown = [v for v in vintages if v not in ALL_VINTAGES]
    if unknown:
        parser.error(f"unknown vintage(s) {unknown}; choose from {ALL_VINTAGES}")

    if not TEMPLATE_PATH.exists():
        print(f"ERROR: template not found at {TEMPLATE_PATH}", file=sys.stderr)
        return 1
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Import huggingface_hub lazily so --dry-run works without auth / the package.
    api = None
    if not args.dry_run:
        from huggingface_hub import HfApi  # noqa: WPS433 (local import by design)

        api = HfApi()
        who = api.whoami()  # fails fast with a clear error if not logged in
        print(f"Authenticated as: {who.get('name', '<unknown>')}\n")

    if args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    for vintage in vintages:
        repo_id = REPO_TEMPLATE.format(vintage=vintage)
        card = render_card(template, vintage)

        if args.dry_run:
            out_path = OUT_DIR / f"{vintage}.md"
            out_path.write_text(card, encoding="utf-8")
            print(f"[{vintage}] rendered -> {out_path}")
            continue

        print(f"[{vintage}] {repo_id}")
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card (replication)",
        )
        print(f"[{vintage}]   uploaded README.md")

        if args.no_publish:
            print(f"[{vintage}]   visibility unchanged (--no-publish)")
        else:
            make_public(api, repo_id)
            print(f"[{vintage}]   set public")

    if args.dry_run:
        print(f"\nDry run complete. Rendered {len(vintages)} card(s) to {OUT_DIR}.")
    else:
        action = "uploaded" if args.no_publish else "uploaded + published"
        print(f"\nDone. {action} {len(vintages)} vintage(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
