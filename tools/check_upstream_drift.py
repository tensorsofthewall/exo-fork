#!/usr/bin/env python3
# pyright: basic
"""Report how far this maintained fork has drifted from its upstream, and whether the drift touches the TBCCL integration. Read-only: it never merges, rebases or moves a tag.

    check_upstream_drift.py --upstream-url https://github.com/exo-explore/exo.git --upstream-branch main \\
                            --base <upstream commit the integration was built on> --tag <immutable integration tag>

It fetches the upstream branch into a throwaway remote ref, then reports: the fork `main` SHA, the upstream SHA, the merge base, how many commits the fork has that
upstream lacks (ahead) and the other way round (behind), how many files upstream changed since the merge base, and the overlap between those files and the
integration touch points. The touch points are derived on every run as the files that differ between --base and --tag (the integration as pinned by the release),
so there is no hand-maintained list to go stale. The answer to "how much work will the next compatibility sync need" is the overlap.

Exit codes follow the maintenance convention: 0 no drift, 5 DRIFT (upstream moved; not a failure), 4 ERROR. Run it inside a full clone of the fork.
"""

import argparse
import os
import subprocess
import sys

OK, ERROR, DRIFT = 0, 4, 5


def git(*args, check=True):
    p = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()[:300]}")
    return p.stdout.strip()


def lines(text):
    return [line for line in text.splitlines() if line]


def analyse(fork_ref, upstream_ref, base, tag):
    merge_base = git("merge-base", fork_ref, upstream_ref)
    ahead = int(git("rev-list", "--count", f"{upstream_ref}..{fork_ref}"))
    behind = int(git("rev-list", "--count", f"{fork_ref}..{upstream_ref}"))
    upstream_files = set(lines(git("diff", "--name-only", merge_base, upstream_ref)))
    touch = set(lines(git("diff", "--name-only", base, tag)))
    return {
        "fork_sha": git("rev-parse", fork_ref),
        "upstream_sha": git("rev-parse", upstream_ref),
        "merge_base": merge_base,
        "ahead": ahead,
        "behind": behind,
        "upstream_files": sorted(upstream_files),
        "touch_points": sorted(touch),
        "overlap": sorted(upstream_files & touch),
    }


def render(r, args):
    out = [
        f"### Upstream drift: {args.name}",
        "",
        "| | |",
        "|---|---|",
        f"| fork main | `{r['fork_sha']}` |",
        f"| upstream {args.upstream_branch} | `{r['upstream_sha']}` |",
        f"| merge base | `{r['merge_base']}` |",
        f"| ahead (fork-only commits) | {r['ahead']} |",
        f"| behind (upstream commits not in the fork) | {r['behind']} |",
        f"| files upstream changed since the merge base | {len(r['upstream_files'])} |",
        f"| integration touch points ({args.base[:12]}..{args.tag}) | {len(r['touch_points'])} files |",
        f"| **upstream changes that touch the integration** | **{len(r['overlap'])} files** |",
        "",
    ]
    if r["overlap"]:
        out += [
            "Files changed upstream that the integration also touches (these are where the next sync will need attention):",
            "",
        ]
        out += [f"- `{f}`" for f in r["overlap"][:60]]
        if len(r["overlap"]) > 60:
            out.append(f"- ... and {len(r['overlap']) - 60} more")
    elif r["behind"]:
        out.append(
            "Upstream moved, but none of the changed files is an integration touch point."
        )
    else:
        out.append("The fork contains everything upstream has.")
    out += ["", "No merge, rebase, tag or pin was changed. This is a monitor."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", default=os.path.basename(os.getcwd()))
    ap.add_argument("--upstream-url", required=True)
    ap.add_argument("--upstream-branch", default="main")
    ap.add_argument(
        "--base",
        required=True,
        help="the upstream commit the integration tag was built on",
    )
    ap.add_argument("--tag", required=True, help="the immutable integration tag")
    ap.add_argument("--fork-ref", default="origin/main")
    args = ap.parse_args()
    try:
        git(
            "fetch",
            "--quiet",
            "--no-tags",
            args.upstream_url,
            f"+refs/heads/{args.upstream_branch}:refs/remotes/drift-upstream/{args.upstream_branch}",
        )
        git(
            "fetch",
            "--quiet",
            "--no-tags",
            "origin",
            f"+refs/tags/{args.tag}:refs/tags/{args.tag}",
            check=False,
        )
        for sha in (args.base, args.tag):
            git("cat-file", "-e", f"{sha}^{{commit}}")
        result = analyse(
            args.fork_ref,
            f"refs/remotes/drift-upstream/{args.upstream_branch}",
            args.base,
            args.tag,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}")
        return ERROR
    text = render(result, args)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    if result["behind"]:
        if os.environ.get("GITHUB_ACTIONS"):
            print(
                f"::warning title=DRIFT::{args.name}: upstream is {result['behind']} commits ahead; {len(result['overlap'])} changed files overlap the integration"
            )
        return DRIFT
    return OK


if __name__ == "__main__":
    sys.exit(main())
