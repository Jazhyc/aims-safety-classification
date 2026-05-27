"""Plotly helpers for 2D embedding scatter plots with hover-text tooltips."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Sequence

import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go


_HOVER_TEMPLATE_BASE = (
    "<b>%{customdata[4]}</b>"
    "<br>dataset: %{customdata[5]}  ·  pred: %{customdata[2]}  ·  true: %{customdata[3]}"
    "<br><b>Intent:</b><br>%{customdata[0]}"
    "<br><b>Prompt:</b><br>%{customdata[1]}"
    "<extra></extra>"
)
# When cluster_display is present in the df it's always useful in hover (so you
# can see which cluster a point belongs to no matter what colouring is active).
_HOVER_TEMPLATE_WITH_CLUSTER = (
    "<b>%{customdata[4]}</b>"
    "<br>cluster: %{customdata[6]}"
    "<br>dataset: %{customdata[5]}  ·  pred: %{customdata[2]}  ·  true: %{customdata[3]}"
    "<br><b>Intent:</b><br>%{customdata[0]}"
    "<br><b>Prompt:</b><br>%{customdata[1]}"
    "<extra></extra>"
)


def _truncate_label(value: object, max_chars: int | None) -> str:
    s = str(value)
    if max_chars is None or len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def _wrap_for_hover(text, width: int = 70, max_chars: int = 400) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    wrapped = "<br>".join(textwrap.wrap(text, width=width))
    return wrapped or text


def interactive_umap_plot_toggleable(
    df: pd.DataFrame,
    color_cols: Sequence[str] = ("dataset", "true_harm_binary", "predicted_harm"),
    *,
    x_col: str = "umap1",
    y_col: str = "umap2",
    intent_col: str = "intent",
    prompt_col: str = "prompt",
    title: str | None = None,
    width: int = 950,
    height: int = 700,
    marker_size: int = 5,
    marker_opacity: float = 0.6,
    legend_top_n: int | None = None,
    legend_label_max_chars: int | None = 40,
) -> "go.Figure":
    """Build a single plotly figure with a dropdown to switch between colorings.

    For each name in `color_cols`, one trace per unique value is added; only the
    first coloring is visible initially. The dropdown swaps which set of traces
    is shown by toggling their `visible` flag — no re-render needed.

    Hover template assumes `customdata = [intent_wrapped, prompt_wrapped,
    predicted_harm, true_harm_binary]`, so `df` must contain `predicted_harm`
    and `true_harm_binary` columns (use empty-string fillers if not applicable).

    `legend_label_max_chars` truncates each legend entry with an ellipsis so a
    long category name (e.g. an LLM cluster label) doesn't steal horizontal
    space from the plot. Hover always shows the full, untruncated name.
    `legend_top_n` separately caps the *number* of legend entries shown (largest
    groups by row count) — leave as `None` unless you have hundreds of groups.
    """
    import plotly.graph_objects as go  # lazy
    import plotly.express as px

    missing = [c for c in color_cols if c not in df.columns]
    if missing:
        raise KeyError(f"color_cols not in DataFrame: {missing}")
    for c in (intent_col, prompt_col, "predicted_harm", "true_harm_binary", "dataset"):
        if c not in df.columns:
            raise KeyError(f"required column {c!r} not in DataFrame")

    plot_df = df.copy()
    plot_df["_intent_wrapped"] = plot_df[intent_col].map(_wrap_for_hover)
    plot_df["_prompt_wrapped"] = plot_df[prompt_col].map(_wrap_for_hover)

    has_cluster_col = "cluster_display" in plot_df.columns
    hover_template = (
        _HOVER_TEMPLATE_WITH_CLUSTER if has_cluster_col else _HOVER_TEMPLATE_BASE
    )

    # Light24 carries us through 24 distinct hues before cycling — enough for
    # cluster-style colorings without making everything look the same.
    palette = px.colors.qualitative.Light24
    fig = go.Figure()
    # (color_col, [trace_index, ...], [show_in_legend, ...]) so the dropdown
    # can restore both visibility AND per-trace legend membership when switching.
    trace_groups: list[tuple[str, list[int], list[bool]]] = []

    for color_col in color_cols:
        values = plot_df[color_col]
        if values.dtype == object or pd.api.types.is_categorical_dtype(values):
            values = values.fillna("(missing)")
        unique_values = values.unique()
        # Numeric-aware sort so "10" comes after "2" and noise (-1) comes first.
        try:
            categories = sorted(unique_values, key=lambda v: (float(v),))
        except (TypeError, ValueError):
            categories = sorted(unique_values, key=lambda v: str(v))

        # Compute which categories are big enough to merit a legend slot.
        # All categories still get a trace and are drawn — `legend_top_n`
        # just controls the legend, not the scatter.
        if legend_top_n is not None and len(categories) > legend_top_n:
            sizes = values.value_counts()
            top_n_cats = set(sizes.head(legend_top_n).index)
        else:
            top_n_cats = set(categories)

        idx_in_group: list[int] = []
        legend_mask: list[bool] = []
        is_first_group = not trace_groups
        import numpy as np

        for i, cat in enumerate(categories):
            mask = (values == cat).to_numpy()
            sub = plot_df.loc[mask]
            # customdata: [intent, prompt, pred_harm, true_harm, full_name,
            #              dataset, (cluster_display if present)]
            # The full name in customdata[4] is what hover shows, so the trace
            # `name` can be truncated for the legend without losing hover info.
            base_customdata = sub[
                ["_intent_wrapped", "_prompt_wrapped", "predicted_harm", "true_harm_binary"]
            ].to_numpy()
            full_name = str(cat)
            full_name_col = np.full((len(sub), 1), full_name, dtype=object)
            dataset_col = sub[["dataset"]].to_numpy()
            stack = [base_customdata, full_name_col, dataset_col]
            if has_cluster_col:
                stack.append(sub[["cluster_display"]].to_numpy())
            customdata = np.hstack(stack)

            in_legend = cat in top_n_cats
            legend_name = _truncate_label(cat, legend_label_max_chars)

            fig.add_trace(
                go.Scattergl(
                    x=sub[x_col],
                    y=sub[y_col],
                    mode="markers",
                    name=legend_name,
                    legendgroup=color_col,
                    marker=dict(
                        size=marker_size,
                        opacity=marker_opacity,
                        color=palette[i % len(palette)],
                    ),
                    customdata=customdata,
                    hovertemplate=hover_template,
                    visible=is_first_group,
                    showlegend=is_first_group and in_legend,
                )
            )
            idx_in_group.append(len(fig.data) - 1)
            legend_mask.append(in_legend)

        trace_groups.append((color_col, idx_in_group, legend_mask))

    n_total = len(fig.data)
    buttons = []
    for color_col, idx_in_group, legend_mask in trace_groups:
        visibility = [False] * n_total
        showlegend = [False] * n_total
        for i, in_legend in zip(idx_in_group, legend_mask):
            visibility[i] = True
            showlegend[i] = in_legend
        buttons.append(
            dict(
                method="update",
                label=color_col,
                args=[
                    {"visible": visibility, "showlegend": showlegend},
                    {"legend.title.text": color_col},
                ],
            )
        )

    # Render the title as an annotation rather than the chart title so it sits
    # in a deterministic place above the dropdown — plotly's auto-positioned
    # title kept overlapping the updatemenu otherwise.
    title_text = title or "Intent embeddings — UMAP (toggle coloring via dropdown)"

    fig.update_layout(
        title=None,
        width=width,
        height=height,
        margin=dict(t=110),  # reserves room for title (top) + dropdown row
        xaxis_title=x_col.upper().replace("UMAP", "UMAP-"),
        yaxis_title=y_col.upper().replace("UMAP", "UMAP-"),
        legend=dict(itemsizing="constant", title=color_cols[0]),
        hoverlabel=dict(align="left", bgcolor="white", font=dict(size=11)),
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                # Bottom-left, just above the plot — well clear of the title
                # (which sits higher up via the annotation below).
                x=0.0,
                y=1.04,
                xanchor="left",
                yanchor="bottom",
                showactive=True,
                pad=dict(r=10, t=5),
            )
        ],
        annotations=[
            # Title — centered above the dropdown row.
            dict(
                text=f"<b>{title_text}</b>",
                x=0.5,
                y=1.20,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=15),
            ),
            # "color by:" label sits to the left of the dropdown.
            dict(
                text="color by:",
                x=0.0,
                y=1.13,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=12),
            ),
        ],
    )
    return fig
