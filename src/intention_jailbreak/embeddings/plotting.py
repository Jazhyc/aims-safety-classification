"""Plotly helpers for 2D embedding scatter plots with hover-text tooltips."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Sequence

import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go


_HOVER_TEMPLATE = (
    "<b>%{fullData.name}</b>"
    "  ·  pred: %{customdata[2]}  ·  true: %{customdata[3]}"
    "<br><b>Intent:</b><br>%{customdata[0]}"
    "<br><b>Prompt:</b><br>%{customdata[1]}"
    "<extra></extra>"
)


def _wrap_for_hover(text, width: int = 70, max_chars: int = 400) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    wrapped = "<br>".join(textwrap.wrap(text, width=width))
    return wrapped or text


def interactive_umap_plot(
    df: pd.DataFrame,
    *,
    color_col: str = "dataset",
    x_col: str = "umap1",
    y_col: str = "umap2",
    intent_col: str = "intent",
    prompt_col: str = "prompt",
    extra_hover_cols: tuple[str, ...] = ("predicted_harm", "true_harm_binary"),
    title: str | None = None,
    width: int = 950,
    height: int = 700,
    marker_size: int = 5,
    marker_opacity: float = 0.6,
) -> "go.Figure":
    """Build an interactive plotly UMAP scatter coloured by `color_col`.

    Hover text shows the wrapped intent + prompt and any `extra_hover_cols`.
    Categorical color columns are coloured with the qualitative palette;
    plotly express auto-switches to a continuous scale for numeric columns.

    Returns the figure so callers can `.show()` or further customise it.
    """
    import plotly.express as px  # lazy: heavy import

    if color_col not in df.columns:
        raise KeyError(f"color_col {color_col!r} not in DataFrame columns")

    plot_df = df.copy()
    plot_df["_intent_wrapped"] = plot_df.get(intent_col, "").map(_wrap_for_hover)
    plot_df["_prompt_wrapped"] = plot_df.get(prompt_col, "").map(_wrap_for_hover)

    # Categorical Nones / NaNs become their own legend entry — replace with a
    # readable placeholder so the legend doesn't show "nan".
    if plot_df[color_col].dtype == object or pd.api.types.is_categorical_dtype(plot_df[color_col]):
        plot_df[color_col] = plot_df[color_col].fillna("(missing)")

    custom_cols = ["_intent_wrapped", "_prompt_wrapped", *extra_hover_cols]
    # Drop any extra_hover_cols that aren't present
    custom_cols = [c for c in custom_cols if c in plot_df.columns]

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        custom_data=custom_cols,
        title=title or f"Intent embeddings — UMAP, coloured by {color_col}",
        width=width,
        height=height,
        render_mode="webgl",
    )

    extras_template = "".join(
        f"  ·  {c}: %{{customdata[{i + 2}]}}"
        for i, c in enumerate(custom_cols[2:])
    )
    hovertemplate = (
        f"<b>%{{fullData.name}}</b>{extras_template}"
        "<br><b>Intent:</b><br>%{customdata[0]}"
        "<br><b>Prompt:</b><br>%{customdata[1]}"
        "<extra></extra>"
    )

    fig.update_traces(
        marker=dict(size=marker_size, opacity=marker_opacity),
        hovertemplate=hovertemplate,
    )
    fig.update_layout(
        legend=dict(itemsizing="constant", title=color_col),
        hoverlabel=dict(align="left", bgcolor="white", font=dict(size=11)),
        xaxis_title=x_col.upper().replace("UMAP", "UMAP-"),
        yaxis_title=y_col.upper().replace("UMAP", "UMAP-"),
    )
    return fig


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
) -> "go.Figure":
    """Build a single plotly figure with a dropdown to switch between colorings.

    For each name in `color_cols`, one trace per unique value is added; only the
    first coloring is visible initially. The dropdown swaps which set of traces
    is shown by toggling their `visible` flag — no re-render needed.

    Hover template assumes `customdata = [intent_wrapped, prompt_wrapped,
    predicted_harm, true_harm_binary]`, so `df` must contain `predicted_harm`
    and `true_harm_binary` columns (use empty-string fillers if not applicable).
    """
    import plotly.graph_objects as go  # lazy
    import plotly.express as px

    missing = [c for c in color_cols if c not in df.columns]
    if missing:
        raise KeyError(f"color_cols not in DataFrame: {missing}")
    for c in (intent_col, prompt_col, "predicted_harm", "true_harm_binary"):
        if c not in df.columns:
            raise KeyError(f"required column {c!r} not in DataFrame")

    plot_df = df.copy()
    plot_df["_intent_wrapped"] = plot_df[intent_col].map(_wrap_for_hover)
    plot_df["_prompt_wrapped"] = plot_df[prompt_col].map(_wrap_for_hover)

    palette = px.colors.qualitative.Plotly
    fig = go.Figure()
    # (color_col, [trace_index, ...]) so the dropdown knows what to toggle.
    trace_groups: list[tuple[str, list[int]]] = []

    for color_col in color_cols:
        values = plot_df[color_col]
        if values.dtype == object or pd.api.types.is_categorical_dtype(values):
            values = values.fillna("(missing)")
        categories = sorted(values.unique(), key=lambda v: str(v))

        idx_in_group: list[int] = []
        is_first_group = not trace_groups
        for i, cat in enumerate(categories):
            mask = (values == cat).to_numpy()
            sub = plot_df.loc[mask]
            customdata = sub[
                ["_intent_wrapped", "_prompt_wrapped", "predicted_harm", "true_harm_binary"]
            ].to_numpy()

            fig.add_trace(
                go.Scattergl(
                    x=sub[x_col],
                    y=sub[y_col],
                    mode="markers",
                    name=str(cat),
                    legendgroup=color_col,
                    marker=dict(
                        size=marker_size,
                        opacity=marker_opacity,
                        color=palette[i % len(palette)],
                    ),
                    customdata=customdata,
                    hovertemplate=_HOVER_TEMPLATE,
                    visible=is_first_group,
                    showlegend=is_first_group,
                )
            )
            idx_in_group.append(len(fig.data) - 1)

        trace_groups.append((color_col, idx_in_group))

    n_total = len(fig.data)
    buttons = []
    for color_col, idx_in_group in trace_groups:
        visibility = [False] * n_total
        for i in idx_in_group:
            visibility[i] = True
        buttons.append(
            dict(
                method="update",
                label=color_col,
                args=[
                    {"visible": visibility, "showlegend": visibility},
                    {"legend.title.text": color_col},
                ],
            )
        )

    fig.update_layout(
        title=title or "Intent embeddings — UMAP (toggle coloring via dropdown)",
        width=width,
        height=height,
        xaxis_title=x_col.upper().replace("UMAP", "UMAP-"),
        yaxis_title=y_col.upper().replace("UMAP", "UMAP-"),
        legend=dict(itemsizing="constant", title=color_cols[0]),
        hoverlabel=dict(align="left", bgcolor="white", font=dict(size=11)),
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=1.0,
                y=1.12,
                xanchor="right",
                yanchor="top",
                showactive=True,
                pad=dict(r=10, t=5),
            )
        ],
        annotations=[
            dict(
                text="color by:",
                x=0.78,
                y=1.10,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )
    return fig
