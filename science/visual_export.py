"""Publication-exportable numeric UMAP figures; UI provides full Chinese labels."""

from threading import Lock

PLOT_LOCK = Lock()


def export_plots(result, out):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "图表导出需安装 requirements-analysis.txt 中的 matplotlib。"
        ) from exc
    with PLOT_LOCK:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, channel in zip(axes.flat, result["channels"]):
            points = channel["points"]
            colors = [p.get("cluster_id", 0) for p in points]
            ax.scatter(
                [p["x"] for p in points],
                [p["y"] for p in points],
                c=colors,
                cmap="tab20",
                s=12,
                alpha=0.45,
            )
            for index, state in enumerate(channel.get("states", [])):
                color = ["#d7792a", "#286ec4"][index]
                refs = state["references"]
                ax.scatter(
                    [r["x"] for r in refs],
                    [r["y"] for r in refs],
                    color=color,
                    marker="o",
                    facecolors="none",
                    s=70,
                    linewidths=1.4,
                )
                ax.scatter(
                    [state["x"]],
                    [state["y"]],
                    color=color,
                    marker="*" if index == 0 else "D",
                    s=180,
                    edgecolors="black",
                    linewidths=0.5,
                    label=state["sample_id"],
                )
            ax.set_title(f"CH {channel['channel']}")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.grid(alpha=0.15)
            if channel.get("states"):
                ax.legend(fontsize=8)
        fig.suptitle(
            "Synthetic demonstration"
            if result["mode"] == "demo"
            else "Six-channel mechanism space"
        )
        fig.tight_layout()
        for suffix in ["png", "svg", "pdf"]:
            fig.savefig(
                out / f"six_channel_umap.{suffix}", dpi=300, bbox_inches="tight"
            )
        plt.close(fig)
        global_points = result.get("global", {}).get("points", [])
        if global_points:
            fig, ax = plt.subplots(figsize=(8, 6))
            for ch in range(1, 7):
                sub = [p for p in global_points if p["channel"] == ch]
                ax.scatter(
                    [p["x"] for p in sub],
                    [p["y"] for p in sub],
                    s=15,
                    alpha=0.7,
                    label=f"CH {ch}",
                )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend()
            fig.tight_layout()
            for suffix in ["png", "svg", "pdf"]:
                fig.savefig(
                    out / f"global_semantic_umap.{suffix}", dpi=300, bbox_inches="tight"
                )
            plt.close(fig)
