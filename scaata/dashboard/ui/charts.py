"""Plotly figures for the HANSEI dashboard. Each takes a frame from
`ui.data` and returns a figure; none touches Streamlit or the network."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scaata.dashboard.ui import theme as T
from scaata.dashboard.ui.data import ADVISORS, short

CONFIG = {"displayModeBar": False, "scrollZoom": False}


def _layout(fig: go.Figure, height: int = 360, legend: bool = True, unified: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=18, t=14, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=T.FONT, color=T.SOFT, size=12),
        hovermode="x unified" if unified else "closest",
        hoverlabel=dict(bgcolor=T.RAISED, bordercolor=T.LINE, font=dict(family=T.MONO, color=T.INK, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=T.SOFT)),
        dragmode=False)
    fig.update_xaxes(showgrid=False, linecolor=T.LINE, tickcolor=T.LINE, tickfont=dict(color=T.MUTED, family=T.MONO,
                     size=11), zeroline=False, showspikes=True, spikecolor=T.MUTED, spikethickness=1,
                     spikedash="dot", spikemode="across")
    fig.update_yaxes(gridcolor="#143238", linecolor="rgba(0,0,0,0)", tickfont=dict(color=T.MUTED, family=T.MONO,
                     size=11), zeroline=False)
    return fig


def empty(message: str, height: int = 220) -> go.Figure:
    fig = _layout(go.Figure(), height, legend=False)
    fig.add_annotation(text=message, showarrow=False, font=dict(color=T.MUTED, size=13))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig


def _end_label(fig: go.Figure, x, y, text: str, colour: str, row: int | None = None) -> None:
    kwargs = dict(row=row, col=1) if row else {}
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=8,
                       font=dict(color=colour, size=12, family=T.FONT), **kwargs)


def race(equity: pd.DataFrame, trades: pd.DataFrame, mode: str = "value") -> go.Figure:
    """HANSEI against just holding. `mode`: "value" (rupees), "gap" (percentage points
    of the starting ₹10,000) or "drawdown" (fall from each one's own peak)."""
    if len(equity) < 2:
        return empty("The chart appears after the second session.")
    fig = go.Figure()
    x = equity["date"]
    if mode == "gap":
        gap = equity["gap"] * 100
        fig.add_trace(go.Scatter(x=x, y=gap.clip(lower=0), mode="lines", line=dict(width=0), fill="tozeroy",
                                 fillcolor="rgba(47,155,184,.28)", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=gap.clip(upper=0), mode="lines", line=dict(width=0), fill="tozeroy",
                                 fillcolor="rgba(229,96,79,.28)", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=gap, mode="lines", name="HANSEI minus holding",
                                 line=dict(color=T.INK, width=2), hovertemplate="%{y:+.2f} pts<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=T.MUTED, width=1))
        fig.update_yaxes(ticksuffix=" pts")
        return _layout(fig, 380, legend=False)
    if mode == "drawdown":
        for column, name, colour in (("hold_dd", "Just holding", T.HOLD), ("hansei_dd", "HANSEI", T.HANSEI)):
            fig.add_trace(go.Scatter(x=x, y=equity[column] * 100, mode="lines", name=name,
                                     line=dict(color=colour, width=2), fill="tozeroy",
                                     fillcolor=_alpha(colour, .16),
                                     hovertemplate=name + " %{y:.2f}%<extra></extra>"))
        fig.update_yaxes(ticksuffix="%")
        return _layout(fig, 380)
    for column, name, colour in (("hold", "Just holding", T.HOLD), ("hansei", "HANSEI", T.HANSEI)):
        fig.add_trace(go.Scatter(x=x, y=equity[column], mode="lines", name=name, line=dict(color=colour, width=2.4),
                                 hovertemplate=name + " ₹%{y:,.0f}<extra></extra>"))
        _end_label(fig, x.iloc[-1], equity[column].iloc[-1], name, colour)
    if len(trades):
        level = equity.set_index("date")["hansei"]
        marks = trades[trades["date"].isin(level.index)]
        for action, symbol in (("BUY", "triangle-up"), ("SELL", "triangle-down")):
            part = marks[marks["action"] == action]
            if part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=part["date"], y=level.loc[part["date"]].values, mode="markers", name=action.title(),
                marker=dict(symbol=symbol, size=11, color=T.ACTION_COLOURS[action], line=dict(color=T.BG, width=2)),
                customdata=part[["symbol", "qty", "price"]].assign(symbol=part["symbol"].map(short)).values,
                hovertemplate=action.title() + " %{customdata[1]} × %{customdata[0]} @ ₹%{customdata[2]:,.2f}"
                              "<extra></extra>"))
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    fig.update_layout(margin=dict(r=96))
    return _layout(fig, 380).update_layout(margin=dict(l=8, r=96, t=14, b=8))


def _alpha(colour: str, alpha: float) -> str:
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def waterfall(story: dict) -> go.Figure:
    """Holding's return, then what being under-invested cost, then what stock choice added."""
    steps = [("Just holding", story["hold"], "absolute"), ("Not fully invested", story["exposure"], "relative"),
             ("Which stocks it held", story["selection"], "relative"), ("HANSEI", story["hansei"], "total")]
    fig = go.Figure(go.Waterfall(
        x=[s[0] for s in steps], y=[s[1] * 100 for s in steps], measure=[s[2] for s in steps],
        text=[f"{s[1]:+.2%}" for s in steps], textposition="outside", textfont=dict(family=T.MONO, color=T.INK),
        connector=dict(line=dict(color=T.LINE, width=1)),
        increasing=dict(marker=dict(color=T.UP)), decreasing=dict(marker=dict(color=T.DOWN)),
        totals=dict(marker=dict(color=T.HANSEI)), width=0.5,
        hovertemplate="%{x}: %{y:+.2f}%<extra></extra>", cliponaxis=False))
    fig.update_yaxes(ticksuffix="%")
    fig = _layout(fig, 320, legend=False, unified=False)
    fig.update_xaxes(showspikes=False)
    return fig


def allocation_bar(parts: list[tuple[str, float]]) -> go.Figure:
    total = sum(v for _, v in parts) or 1.0
    fig = go.Figure()
    shades = ["#2F9BB8", "#3FA9C5", "#57B6CF", "#74C3D8", "#93D0E1"]
    for i, (label, value) in enumerate(parts):
        cash = label == "Liquid fund"
        colour = "#3B5257" if cash else shades[i % len(shades)]
        fig.add_trace(go.Bar(y=[""], x=[value], orientation="h", name=label,
                             marker=dict(color=colour, line=dict(color=T.PANEL, width=2)),
                             text=f"{label}<br>{value / total:.0%}", textposition="inside", textangle=0, insidetextanchor="middle",
                             textfont=dict(color=T.INK if cash else "#04161A", size=12, family=T.FONT),
                             hovertemplate=f"{label}: ₹%{{x:,.0f}} ({value / total:.1%})<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig = _layout(fig, 120, legend=False, unified=False)
    fig.update_xaxes(visible=False, showspikes=False); fig.update_yaxes(visible=False)
    return fig


def exposure_map(decisions: pd.DataFrame) -> go.Figure:
    """Stock × session: how much of its share HANSEI held, with its trades marked."""
    if decisions.empty:
        return empty("No decisions yet.")
    grid = decisions.pivot(index="symbol", columns="session", values="held")
    reasons = decisions.pivot(index="symbol", columns="session", values="reason")
    labels = [short(s) for s in grid.index]
    x = [d.strftime("%d %b") for d in grid.columns]
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=x, y=labels, zmin=0, zmax=1.2, colorscale=T.SEQUENTIAL, xgap=2, ygap=2,
        customdata=reasons.values, hovertemplate="%{y} · %{x}<br>held %{z:.0%} of its share<br>%{customdata}"
                                                 "<extra></extra>",
        colorbar=dict(title=dict(text="held", font=dict(color=T.MUTED, size=11)), tickformat=".0%", thickness=10,
                      len=0.9, tickfont=dict(color=T.MUTED, family=T.MONO, size=10), outlinewidth=0)))
    acted = decisions[decisions["action"] != "HOLD"]
    for action, symbol in (("BUY", "triangle-up"), ("SELL", "triangle-down"), ("REBALANCE", "diamond")):
        part = acted[acted["action"] == action]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(x=[d.strftime("%d %b") for d in part["session"]], y=part["symbol"].map(short),
                                 mode="markers", name=action.title(), hoverinfo="skip",
                                 marker=dict(symbol=symbol, size=9, color=T.INK, line=dict(color=T.BG, width=1))))
    fig = _layout(fig, 300, unified=False)
    fig.update_xaxes(showspikes=False, type="category", nticks=12); fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def views_map(decisions: pd.DataFrame, symbol: str) -> go.Figure:
    """One stock: each advisor's view on every session, −1 (get out) to +1 (own it)."""
    part = decisions[decisions["symbol"] == symbol].sort_values("session")
    if part.empty:
        return empty("No views recorded for this stock yet.")
    x = [d.strftime("%d %b") for d in part["session"]]
    fig = go.Figure(go.Heatmap(
        z=[part[a].values for a in ADVISORS], x=x, y=[T.ADVISOR_LABELS[a] for a in ADVISORS], zmin=-1, zmax=1,
        colorscale=T.DIVERGING, xgap=2, ygap=2, hovertemplate="%{y} · %{x}<br>view %{z:+.2f}<extra></extra>",
        colorbar=dict(tickvals=[-1, 0, 1], ticktext=["get out", "no view", "own it"], thickness=10, len=0.9,
                      tickfont=dict(color=T.MUTED, size=10), outlinewidth=0)))
    fig = _layout(fig, 230, legend=False, unified=False)
    fig.update_xaxes(showspikes=False, type="category", nticks=12); fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def trust_lines(trust: pd.DataFrame, lessons: pd.DataFrame) -> go.Figure:
    if trust.empty:
        return empty("Trust is equal (25% each) until the first lesson.")
    fig = go.Figure()
    fig.add_hline(y=25, line=dict(color=T.MUTED, width=1, dash="dot"),
                  annotation_text="equal trust · 25%", annotation_position="bottom left",
                  annotation_font=dict(color=T.MUTED, size=11))
    for advisor in ADVISORS:
        colour = T.ADVISOR_COLOURS[advisor]
        fig.add_trace(go.Scatter(x=trust["session"], y=trust[advisor] * 100, mode="lines",
                                 name=T.ADVISOR_LABELS[advisor], line=dict(color=colour, width=2.4),
                                 hovertemplate=T.ADVISOR_LABELS[advisor] + " %{y:.1f}%<extra></extra>"))
        _end_label(fig, trust["session"].iloc[-1], trust[advisor].iloc[-1] * 100, T.ADVISOR_LABELS[advisor], colour)
    if len(lessons):
        first = lessons["learned_on"].min()
        fig.add_vline(x=first, line=dict(color=T.TAN, width=1, dash="dash"))
        fig.add_annotation(x=first, y=1, yref="paper", text="first lesson", showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=T.TAN, size=11))
    fig.update_yaxes(ticksuffix="%")
    return _layout(fig, 360).update_layout(margin=dict(l=8, r=96, t=14, b=8))


def lessons_strip(lessons: pd.DataFrame) -> go.Figure:
    """Each lesson, as how far it moved trust in each advisor."""
    if lessons.empty:
        return empty("No lessons yet: the first judgement comes 20 sessions after the first belief.")
    fig = go.Figure()
    x = [d.strftime("%d %b") for d in lessons["about_day"]]
    for advisor in ADVISORS:
        fig.add_trace(go.Bar(x=x, y=lessons[f"d_{advisor}"] * 100, name=T.ADVISOR_LABELS[advisor],
                             marker=dict(color=T.ADVISOR_COLOURS[advisor], line=dict(color=T.PANEL, width=1)),
                             hovertemplate=T.ADVISOR_LABELS[advisor] + " %{y:+.2f} pts<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.35)
    fig.add_hline(y=0, line=dict(color=T.MUTED, width=1))
    fig.update_yaxes(ticksuffix=" pts")
    fig = _layout(fig, 300)
    fig.update_xaxes(type="category", title=dict(text="the day the judged beliefs were formed",
                                                 font=dict(color=T.MUTED, size=11)))
    return fig


def hit_rates(advisor_scores: dict) -> go.Figure:
    """Share of each advisor's opinions that pointed the right way, at two horizons."""
    fig = go.Figure()
    fig.add_vline(x=50, line=dict(color=T.MUTED, width=1, dash="dot"))
    names = [a for a in (*ADVISORS, "relstr")]
    for horizon, label, symbol in (("5_sessions", "after 5 sessions", "circle"),
                                   ("20_sessions", "after 20 sessions", "diamond")):
        scores = advisor_scores.get(horizon, {})
        rows = [(a, scores[a]) for a in names if scores.get(a, {}).get("hit_rate") is not None]
        if not rows:
            continue
        fig.add_trace(go.Scatter(
            x=[r["hit_rate"] * 100 for _, r in rows], y=[T.ADVISOR_LABELS[a] for a, _ in rows], mode="markers",
            name=label, marker=dict(symbol=symbol, size=13, color=[T.ADVISOR_COLOURS[a] for a, _ in rows],
                                    line=dict(color=T.BG, width=2)),
            customdata=[r["judged"] for _, r in rows],
            hovertemplate="%{y} · " + label + "<br>right %{x:.0f}% of %{customdata} opinions<extra></extra>"))
    fig.update_xaxes(range=[20, 80], ticksuffix="%", showspikes=False)
    fig.update_yaxes(showgrid=True, autorange="reversed")
    fig.add_annotation(x=50, y=1.02, yref="paper", text="coin flip", showarrow=False, yanchor="bottom",
                       font=dict(color=T.MUTED, size=11))
    return _layout(fig, 300, unified=False).update_layout(margin=dict(l=8, r=18, t=44, b=8),
                                                           legend=dict(y=1.12))


def beliefs_map(beliefs: pd.DataFrame) -> go.Figure:
    """Every belief on file: judged ones coloured by what the stock did next, pending ones hollow."""
    if beliefs.empty:
        return empty("No beliefs recorded yet.")
    judged = beliefs[beliefs["judged"]]
    pending = beliefs[~beliefs["judged"]]
    fig = go.Figure()
    if len(judged):
        fig.add_trace(go.Scatter(
            x=judged["date"], y=judged["symbol"].map(short), mode="markers", name="judged",
            marker=dict(symbol="square", size=13, color=judged["forward_return"] * 100, cmin=-15, cmax=15,
                        colorscale=T.DIVERGING,
                        colorbar=dict(title=dict(text="next 20 sessions", font=dict(color=T.MUTED, size=11)),
                                      ticksuffix="%", thickness=10, len=0.9, outlinewidth=0,
                                      tickfont=dict(color=T.MUTED, family=T.MONO, size=10))),
            hovertemplate="%{y} · believed on %{x|%d %b}<br>then %{marker.color:+.1f}% vs a liquid fund"
                          "<extra></extra>"))
    if len(pending):
        fig.add_trace(go.Scatter(
            x=pending["date"], y=pending["symbol"].map(short), mode="markers", name="waiting to be judged",
            marker=dict(symbol="square-open", size=12, color=T.MUTED, line=dict(width=1.5)),
            hovertemplate="%{y} · believed on %{x|%d %b}<br>not yet 20 sessions old<extra></extra>"))
    fig = _layout(fig, 280, unified=False)
    fig.update_xaxes(showspikes=False); fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def news_bubbles(news: pd.DataFrame) -> go.Figure:
    """Every material story: when, about whom, how good or bad, how much it mattered."""
    if news.empty:
        return empty("No material stories recorded yet.")
    frame = news.dropna(subset=["published"])
    wrapped = frame["title"].str.wrap(46).str.replace("\n", "<br>")
    fig = go.Figure(go.Scatter(
        x=frame["published"], y=frame["symbol"].map(short), mode="markers",
        marker=dict(size=(9 + 14 * frame["material"] + 3 * frame["count"].clip(upper=5)).tolist(),
                    color=frame["sentiment"], cmin=-1, cmax=1, colorscale=T.DIVERGING, opacity=0.9,
                    line=dict(color=T.BG, width=2),
                    colorbar=dict(tickvals=[-1, 0, 1], ticktext=["bad", "neutral", "good"], thickness=10, len=0.9,
                                  outlinewidth=0, tickfont=dict(color=T.MUTED, size=10))),
        customdata=list(zip(wrapped, frame["source"], frame["material"], frame["count"])),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{x|%d %b}<br>sentiment %{marker.color:+.2f} · "
                      "materiality %{customdata[2]:.1f} · %{customdata[3]} outlet(s)<extra></extra>"))
    fig = _layout(fig, 380, legend=False, unified=False)
    fig.update_xaxes(showspikes=False); fig.update_yaxes(showgrid=True, autorange="reversed")
    return fig


def news_views(decisions: pd.DataFrame) -> go.Figure:
    """The news advisor's view per stock per session."""
    if decisions.empty:
        return empty("No news views yet.")
    grid = decisions.pivot(index="symbol", columns="session", values="news")
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[d.strftime("%d %b") for d in grid.columns], y=[short(s) for s in grid.index],
        zmin=-1, zmax=1, colorscale=T.DIVERGING, xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}<br>news view %{z:+.2f}<extra></extra>",
        colorbar=dict(tickvals=[-1, 0, 1], ticktext=["bad", "silent", "good"], thickness=10, len=0.9,
                      outlinewidth=0, tickfont=dict(color=T.MUTED, size=10))))
    fig = _layout(fig, 260, legend=False, unified=False)
    fig.update_xaxes(showspikes=False, type="category", nticks=12); fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def backtest(curve: pd.DataFrame) -> go.Figure:
    """Top: both equity curves. Bottom: each one's fall from its own peak. Shared time axis."""
    if curve.empty:
        return empty("Run `python -m scaata.agent.backtest` to produce the replay.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.64, 0.36], vertical_spacing=0.05)
    for column, name, colour in (("hold", "Just holding", T.HOLD), ("hansei", "HANSEI", T.HANSEI)):
        fig.add_trace(go.Scatter(x=curve["date"], y=curve[column], mode="lines", name=name, legendgroup=name,
                                 line=dict(color=colour, width=2),
                                 hovertemplate=name + " ₹%{y:,.0f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=curve["date"], y=curve[f"{column}_dd"] * 100, mode="lines", name=name,
                                 legendgroup=name, showlegend=False, line=dict(color=colour, width=1.6),
                                 fill="tozeroy", fillcolor=_alpha(colour, .16),
                                 hovertemplate=name + " %{y:.1f}% below peak<extra></extra>"), row=2, col=1)
    fig = _layout(fig, 520)
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False))
    fig.update_layout(dragmode="zoom")
    return fig


def drawdown_dumbbell(blocks: pd.DataFrame) -> go.Figure:
    """Worst fall in each multi-year block: HANSEI's dot against holding's."""
    if blocks.empty:
        return empty("Run `python -m scaata.agent.backtest --long` to produce the blocks.")
    fig = go.Figure()
    for _, row in blocks.iterrows():
        fig.add_trace(go.Scatter(x=[row["hold_dd"] * 100, row["hansei_dd"] * 100], y=[row["block"]] * 2, mode="lines",
                                 line=dict(color=T.LINE, width=4), hoverinfo="skip", showlegend=False))
    for column, name, colour in (("hold_dd", "Just holding", T.HOLD), ("hansei_dd", "HANSEI", T.HANSEI)):
        fig.add_trace(go.Scatter(x=blocks[column] * 100, y=blocks["block"], mode="markers", name=name,
                                 marker=dict(size=14, color=colour, line=dict(color=T.BG, width=2)),
                                 hovertemplate=name + " · %{y}<br>worst fall %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(ticksuffix="%", showspikes=False)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 320, unified=False)


def return_bars(blocks: pd.DataFrame) -> go.Figure:
    """Yearly return difference in each block: above zero HANSEI won, below it lost."""
    if blocks.empty:
        return empty("Run `python -m scaata.agent.backtest --long` to produce the blocks.")
    values = blocks["cagr_diff"] * 100
    fig = go.Figure(go.Bar(
        x=blocks["block"], y=values, width=0.5, marker=dict(color=[T.UP if v >= 0 else T.DOWN for v in values]),
        text=[f"{v:+.1f}" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(family=T.MONO, color=T.INK),
        hovertemplate="%{x}<br>HANSEI minus holding: %{y:+.2f}% a year<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T.MUTED, width=1))
    fig.update_yaxes(ticksuffix="%")
    fig = _layout(fig, 320, legend=False, unified=False)
    fig.update_xaxes(showspikes=False, type="category")
    return fig


def knob_grid(grid: list[dict]) -> go.Figure:
    """Sharpe edge over holding for every setting of the two tuning knobs."""
    if not grid:
        return empty("No settings grid in the backtest report.")
    frame = pd.DataFrame(grid)
    table = frame.pivot(index="act_threshold", columns="min_hold_days", values="sharpe_diff")
    bound = max(0.3, float(frame["sharpe_diff"].abs().max()))
    fig = go.Figure(go.Heatmap(
        z=table.values, x=[f"wait {d} sessions" for d in table.columns],
        y=[f"act on a {t:.0%} gap" for t in table.index], zmin=-bound, zmax=bound, colorscale=T.DIVERGING,
        xgap=3, ygap=3, text=[[f"{v:+.2f}" for v in row] for row in table.values], texttemplate="%{text}",
        textfont=dict(family=T.MONO, color=T.INK, size=13), showscale=False,
        hovertemplate="%{y}, %{x}<br>Sharpe edge over holding %{z:+.2f}<extra></extra>"))
    fig = _layout(fig, 260, legend=False, unified=False)
    fig.update_xaxes(showspikes=False, side="bottom"); fig.update_yaxes(showgrid=False)
    return fig


def forest(signals: pd.DataFrame) -> go.Figure:
    """Each advisor's return spread with its 95% interval. An interval crossing zero is not evidence."""
    if signals.empty:
        return empty("Run `python -m scaata.agent.signal_study` to produce the study.")
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=T.MUTED, width=1))
    splits = list(dict.fromkeys(signals["split"]))
    for split, symbol in zip(splits, ("circle", "diamond", "square")):
        part = signals[signals["split"] == split]
        fig.add_trace(go.Scatter(
            x=part["spread"] * 100, y=[T.ADVISOR_LABELS.get(a, a) for a in part["advisor"]], mode="markers",
            name=split, marker=dict(symbol=symbol, size=11, color=[T.ADVISOR_COLOURS.get(a, T.MUTED)
                                                                   for a in part["advisor"]],
                                    line=dict(color=T.BG, width=2)),
            error_x=dict(type="data", symmetric=False, array=(part["high"] - part["spread"]) * 100,
                         arrayminus=(part["spread"] - part["low"]) * 100, color=T.MUTED, thickness=1.4, width=4),
            customdata=list(zip(part["low"] * 100, part["high"] * 100, part["opinions"])),
            hovertemplate="%{y} · " + split + "<br>spread %{x:+.2f}% (95% CI %{customdata[0]:+.2f} to "
                          "%{customdata[1]:+.2f})<br>%{customdata[2]:,} opinions<extra></extra>"))
    fig.update_layout(scattermode="group", scattergap=0.5)
    fig.update_xaxes(ticksuffix="%", showspikes=False)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 300, unified=False)


def market(prices: pd.DataFrame, days: int) -> go.Figure:
    """The five stocks and the Nifty, rebased to 100 at the start of the window."""
    window = prices.tail(days).dropna(how="all")
    if len(window) < 2:
        return empty("Not enough price history.")
    fig = go.Figure()
    shades = ["#2F9BB8", "#3FA873", "#B26BC9", "#C4822B", "#E5604F"]
    stocks = [c for c in window.columns if not str(c).startswith("^")]
    for column in [c for c in window.columns if str(c).startswith("^")] + stocks:
        series = window[column].dropna()
        if series.empty:
            continue
        rebased = series / series.iloc[0] * 100
        index = str(column).startswith("^")
        colour = T.INK if index else shades[stocks.index(column) % len(shades)]
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased, mode="lines", name=short(column),
                                 line=dict(color=colour, width=1.6 if index else 2, dash="dot" if index else "solid"),
                                 hovertemplate=short(column) + " %{y:.1f}<extra></extra>"))
    fig.add_hline(y=100, line=dict(color=T.LINE, width=1))
    return _layout(fig, 380)


VERDICT_COLOURS = {"supported": "#3FA873", "not_supported": T.DOWN, "inconclusive": T.HOLD,
                   "failed_review": "#B26BC9", "ran": T.MUTED, "registered": T.MUTED}
VERDICT_WORDS = {"supported": "Supported", "not_supported": "Not supported", "inconclusive": "Inconclusive",
                 "failed_review": "Failed review", "ran": "Awaiting review", "registered": "Filed, not run"}


def evidence_map(records: list[dict]) -> go.Figure:
    """Each Sharpe-based experiment as a dot: its effect in the tuning years against its effect
    in the years the rule was never tuned on. Only the top-right or bottom-left corner is a
    consistent result."""
    rows = []
    for r in records:
        primary = (r.get("results") or {}).get("primary") or {}
        values = primary.get("values") or {}
        if "tune" in values and "holdout" in values and primary.get("label", "").startswith("Sharpe"):
            verdict = (r.get("review") or {}).get("verdict", "ran")
            rows.append((r["title"], values["tune"], values["holdout"], verdict, primary["label"]))
    if not rows:
        return empty("No experiments with a result in both spans yet.")
    bound = max(0.25, max(max(abs(x), abs(y)) for _, x, y, _, _ in rows) * 1.35)
    fig = go.Figure()
    for x0, x1, y0, y1 in ((0, bound, 0, bound), (-bound, 0, -bound, 0)):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor="rgba(47,155,184,.07)", line=dict(width=0),
                      layer="below")
    fig.add_hline(y=0, line=dict(color=T.MUTED, width=1)); fig.add_vline(x=0, line=dict(color=T.MUTED, width=1))
    for text, x, y, ax, ay in (("better in both spans", bound, bound, "right", "top"),
                               ("worse in both spans", -bound, -bound, "left", "bottom"),
                               ("only in the tuning years", bound, -bound, "right", "bottom"),
                               ("only in the later years", -bound, bound, "left", "top")):
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor=ax, yanchor=ay,
                           font=dict(color=T.MUTED, size=11), xshift=-6 if ax == "right" else 6,
                           yshift=-4 if ay == "top" else 4)
    for verdict in dict.fromkeys(v for *_, v, _ in rows):
        part = [r for r in rows if r[3] == verdict]
        fig.add_trace(go.Scatter(
            x=[r[1] for r in part], y=[r[2] for r in part], mode="markers+text", name=VERDICT_WORDS[verdict],
            text=[r[0] for r in part], textposition="top center", textfont=dict(color=T.SOFT, size=11),
            marker=dict(size=16, color=VERDICT_COLOURS[verdict], line=dict(color=T.BG, width=2)),
            customdata=[r[4] for r in part],
            hovertemplate="<b>%{text}</b><br>%{customdata}<br>2007–2019: %{x:+.3f}<br>2020 on: %{y:+.3f}"
                          "<extra></extra>"))
    fig.update_xaxes(range=[-bound, bound], showspikes=False,
                     title=dict(text="effect in 2007–2019 (the years the rule was tuned on)",
                                font=dict(color=T.MUTED, size=11)))
    fig.update_yaxes(range=[-bound, bound], title=dict(text="effect in 2020 onward (never tuned on)",
                                                       font=dict(color=T.MUTED, size=11)))
    return _layout(fig, 440, unified=False)


def regime_bars(table: list[dict]) -> go.Figure:
    """HANSEI's yearly return minus holding's, inside each market condition, per span."""
    if not table:
        return empty("The market-condition study has not been run.")
    names = {"up": "Basket rising", "down": "Basket falling", "calm": "Calm", "choppy": "Choppy"}
    spans = {"tune": ("2007–2019", "circle", 0.55), "holdout": ("2020 onward", "diamond", 1.0)}
    fig = go.Figure()
    for period, (label, _, opacity) in spans.items():
        part = [r for r in table if r["period"] == period]
        values = [r["difference"] * 100 for r in part]
        fig.add_trace(go.Bar(
            x=[names.get(r["regime"], r["regime"]) for r in part], y=values, name=label, opacity=opacity,
            marker=dict(color=[T.UP if v >= 0 else T.DOWN for v in values], line=dict(color=T.PANEL, width=2)),
            text=[f"{v:+.1f}" for v in values], textposition="outside", cliponaxis=False,
            textfont=dict(family=T.MONO, color=T.INK, size=11),
            customdata=[(r["sessions"], r["hansei"] * 100, r["holding"] * 100) for r in part],
            hovertemplate="%{x} · " + label + "<br>HANSEI %{customdata[1]:+.1f}% a year, holding "
                          "%{customdata[2]:+.1f}%<br>%{customdata[0]:,} sessions<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T.MUTED, width=1))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(ticksuffix="%")
    fig = _layout(fig, 360, unified=False)
    fig.update_xaxes(showspikes=False, type="category")
    return fig


def experiment_curves(curves: dict, period: str, same: bool) -> go.Figure:
    """One experiment's equity curves over one span: the variant, the live rule, and holding."""
    curve = (curves or {}).get(period)
    if not curve:
        return empty("No curve stored for this span.")
    fig = go.Figure()
    series = [("holding", "Just holding", T.HOLD, "solid")]
    series += [("variant", "The live rule", T.HANSEI, "solid")] if same else [
        ("baseline", "The live rule", T.HANSEI, "solid"), ("variant", "This variant", T.INK, "dot")]
    for key, name, colour, dash in series:
        fig.add_trace(go.Scatter(x=pd.to_datetime(curve["dates"]), y=curve[key], mode="lines", name=name,
                                 line=dict(color=colour, width=2, dash=dash),
                                 hovertemplate=name + " ₹%{y:,.0f}<extra></extra>"))
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    return _layout(fig, 300)
