#!/usr/bin/env python3
"""Fetch trending GitHub repositories and plot them as a dark-mode bar chart."""

import sys
import subprocess
from datetime import datetime, timedelta, timezone

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fetch_trending(top_n=10):
    """GitHub has no official 'trending' endpoint, so we approximate it:
    repos created in the last 30 days, sorted by stars (descending)."""
    since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    url = "https://api.github.com/search/repositories"
    params = {
        "q": f"created:>{since}",
        "sort": "stars",
        "order": "desc",
        "per_page": top_n,
    }
    headers = {"Accept": "application/vnd.github+json"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("items", [])

    repos = []
    for it in items[:top_n]:
        repos.append({
            "name": it["full_name"],
            "stars": it["stargazers_count"],
            "language": it.get("language") or "N/A",
        })
    return repos


def plot(repos, outfile="trending.png"):
    plt.style.use("dark_background")

    names = [f'{r["name"]}\n({r["language"]})' for r in repos]
    stars = [r["stars"] for r in repos]

    # Color bars by star count using a vivid colormap
    norm = plt.Normalize(min(stars), max(stars))
    colors = plt.cm.viridis(norm(stars))

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(names[::-1], stars[::-1], color=colors[::-1], edgecolor="none")

    ax.set_title("Top 10 Trending GitHub Repositories (last 30 days)",
                 fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Stars", fontsize=12)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Annotate star counts at the end of each bar
    for bar, s in zip(bars, stars[::-1]):
        ax.text(bar.get_width() + max(stars) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{s:,}", va="center", ha="left", fontsize=9, color="#dddddd")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved chart to {outfile}")
    return outfile


def open_image(path):
    """Open the image with the OS default previewer."""
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", path], check=False)
    elif sys.platform.startswith("win"):
        subprocess.run(["cmd", "/c", "start", "", path], check=False, shell=False)


def main():
    repos = fetch_trending(10)
    if not repos:
        print("No repositories returned from GitHub API.", file=sys.stderr)
        sys.exit(1)

    print("Top trending repositories:")
    for i, r in enumerate(repos, 1):
        print(f"{i:2}. {r['name']:<45} {r['stars']:>7,} stars  [{r['language']}]")

    out = plot(repos)
    open_image(out)


if __name__ == "__main__":
    main()
