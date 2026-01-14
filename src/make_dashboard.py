import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

TOP_GAMES_CSV = os.path.join(OUTPUTS_DIR, "top_50_choke_games.csv")
TEAM_RANK_CSV = os.path.join(OUTPUTS_DIR, "team_choke_rankings.csv")

INDEX_HTML = os.path.join(DOCS_DIR, "index.html")
DASHBOARD_HTML = os.path.join(DOCS_DIR, "dashboard.html")


def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Make sure outputs/*.csv exist (run build_choke_index.py first)."
        )


def to_num(df: pd.DataFrame, col: str, default=0.0) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


def round_bucket(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).lower()
    if "wild" in s:
        return "Wild Card"
    if "div" in s:
        return "Divisional"
    if "conf" in s or "champ" in s:
        return "Conference"
    if "super" in s or "sb" in s:
        return "Super Bowl"
    return "Unknown"


def build_label(row):
    season = safe_int(row.get("season", ""))
    rnd = row.get("round", "")
    team = row.get("team", "")
    opp = row.get("opp", "")
    pf = safe_int(row.get("points_for", ""))
    pa = safe_int(row.get("points_against", ""))
    lead = safe_int(row.get("max_lead", ""))
    return f"{season} {rnd}: {team} vs {opp} ({pf}-{pa}) | Lead {lead}"


def clamp01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def try_load_team_meta():
    """
    Best-effort team metadata (logos + conf). If it fails (no internet), charts still work.
    """
    url = "https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/teams_colors_logos.csv"
    try:
        meta = pd.read_csv(url)
        if "team_abbr" in meta.columns:
            meta = meta.rename(columns={"team_abbr": "team"})
        meta["team"] = meta["team"].astype(str).str.upper()

        if "team_logo_espn" not in meta.columns and "team_logo_wikipedia" in meta.columns:
            meta["team_logo_espn"] = meta["team_logo_wikipedia"]

        keep = [c for c in ["team", "team_conf", "team_logo_espn"] if c in meta.columns]
        return meta[keep].copy()
    except Exception:
        return None


def _lock_plotly_interactions(fig: go.Figure) -> go.Figure:
    """
    Make charts scroll-friendly on mobile by disabling drag interactions and zoom/pan ranges.
    This reduces the "chart steals scrolling" issue.
    """
    fig.update_layout(dragmode=False)
    try:
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
    except Exception:
        pass
    return fig


def main():
    require_file(TOP_GAMES_CSV)
    require_file(TEAM_RANK_CSV)

    games = pd.read_csv(TOP_GAMES_CSV)
    teams = pd.read_csv(TEAM_RANK_CSV)

    # -----------------------
    # Normalize games
    # -----------------------
    for col in ["true_choke_score", "max_lead", "importance_weight", "late_weight", "points_for", "points_against"]:
        games = to_num(games, col, 0.0)

    for col in ["team", "opp", "round"]:
        if col not in games.columns:
            games[col] = ""

    if "label" not in games.columns:
        games["label"] = games.apply(build_label, axis=1)

    games["team"] = games["team"].astype(str).str.upper()
    games["opp"] = games["opp"].astype(str).str.upper()
    games["round_bucket"] = games["round"].apply(round_bucket)

    # Round-unweighted score (removes importance_weight / round impact only)
    iw = pd.to_numeric(games.get("importance_weight", 1.0), errors="coerce").fillna(1.0)
    iw = iw.replace(0, 1.0)
    games["round_unweighted_score"] = games["true_choke_score"] / iw

    games_top50_true = games.sort_values("true_choke_score", ascending=False).head(50).copy()
    games_top50_unw = games.sort_values("round_unweighted_score", ascending=False).head(50).copy()

    # -----------------------
    # Normalize teams
    # -----------------------
    if "total_choke" not in teams.columns and "total_true_choke" in teams.columns:
        teams["total_choke"] = teams["total_true_choke"]
    if "avg_choke_per_loss" not in teams.columns and "avg_true_choke_per_loss" in teams.columns:
        teams["avg_choke_per_loss"] = teams["avg_true_choke_per_loss"]

    if "team" not in teams.columns:
        teams["team"] = ""
    teams["team"] = teams["team"].astype(str).str.upper()

    for col in ["total_choke", "avg_choke_per_loss", "choke_rate", "playoff_losses", "playoff_games"]:
        teams = to_num(teams, col, 0.0)

    # -----------------------
    # Team meta (logos/conf)
    # -----------------------
    meta = try_load_team_meta()
    logo_map, conf_map = {}, {}
    if meta is not None:
        if "team_logo_espn" in meta.columns:
            logo_map = dict(zip(meta["team"], meta["team_logo_espn"]))
        if "team_conf" in meta.columns:
            conf_map = dict(zip(meta["team"], meta["team_conf"]))

    teams["team_conf"] = teams["team"].map(conf_map).fillna("Unknown")
    teams["logo"] = teams["team"].map(logo_map).fillna("")

    # -----------------------
    # Styling + Plotly config
    # -----------------------
    template = "plotly_dark"

    base_layout = dict(
        template=template,
        font=dict(size=14),
        margin=dict(l=18, r=18, t=72, b=18),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
    )

    # Mobile-friendly interactions: no zoom/pan, no scroll hijack
    config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": False,
        "doubleClick": "reset",
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"
        ],
    }

    ROUND_COLORS = {
        "Wild Card": "#60a5fa",
        "Divisional": "#34d399",
        "Conference": "#fbbf24",
        "Super Bowl": "#f87171",
        "Unknown": "#9ca3af",
    }
    CONF_COLORS = {"AFC": "#60a5fa", "NFC": "#34d399", "Unknown": "#9ca3af"}

    # -----------------------
    # CHART 1: Top chokes
    # Desktop: Top 25 horizontal (detailed labels)
    # Mobile: Top 15 vertical by rank (clean, no squeezed labels)
    # -----------------------
    top25 = games_top50_true.head(25).copy()
    top25 = top25.sort_values("true_choke_score", ascending=True).copy()

    fig1_desktop = px.bar(
        top25,
        x="true_choke_score",
        y="label",
        orientation="h",
        color="round_bucket",
        color_discrete_map=ROUND_COLORS,
        template=template,
    )
    fig1_desktop.update_layout(
        **base_layout,
        height=920,
        legend_title_text="Round",
        title=dict(text="Top 25 Biggest Playoff Chokes (True Score)", x=0.02, xanchor="left"),
    )
    fig1_desktop.update_yaxes(title="", automargin=True)
    fig1_desktop.update_xaxes(title="True Choke Score")
    _lock_plotly_interactions(fig1_desktop)

    top15 = games_top50_true.head(15).copy()
    top15 = top15.sort_values("true_choke_score", ascending=False).reset_index(drop=True)
    top15["rank"] = top15.index + 1
    top15["rank_label"] = top15["rank"].astype(str)

    fig1_mobile = px.bar(
        top15,
        x="rank_label",
        y="true_choke_score",
        color="round_bucket",
        color_discrete_map=ROUND_COLORS,
        template=template,
        hover_data={"label": True, "rank_label": False},
    )
    fig1_mobile.update_layout(
        **base_layout,
        height=420,
        legend_title_text="Round",
        title=dict(text="Top 15 Biggest Playoff Chokes (True Score)", x=0.02, xanchor="left"),
        margin=dict(l=10, r=10, t=72, b=32),
    )
    fig1_mobile.update_xaxes(title="Rank", tickangle=0)
    fig1_mobile.update_yaxes(title="True Choke Score")
    _lock_plotly_interactions(fig1_mobile)

    # -----------------------
    # CHART 2: Choke MOST (mobile tweaks)
    # -----------------------
    most_df = teams.sort_values("choke_rate", ascending=False).head(16).copy()

    fig2 = px.bar(
        most_df.iloc[::-1],
        x="choke_rate",
        y="team",
        orientation="h",
        color="team_conf",
        color_discrete_map=CONF_COLORS,
        template=template,
    )
    fig2.update_layout(
        **base_layout,
        height=520,
        legend_title_text="Conf",
        title=dict(text="Teams That Choke MOST (Rate)", x=0.02, xanchor="left"),
    )
    fig2.update_yaxes(title="", automargin=True)
    fig2.update_xaxes(title="Choke Rate (10+ leads blown per playoff loss)")
    _lock_plotly_interactions(fig2)

    fig2_mobile = fig2.to_dict()
    fig2_mobile = go.Figure(fig2_mobile)
    fig2_mobile.update_layout(height=420, margin=dict(l=10, r=10, t=72, b=18))
    _lock_plotly_interactions(fig2_mobile)

    # -----------------------
    # CHART 3: Choke WORST (mobile tweaks)
    # -----------------------
    worst_df = teams.sort_values("avg_choke_per_loss", ascending=False).head(16).copy()

    fig3 = px.bar(
        worst_df.iloc[::-1],
        x="avg_choke_per_loss",
        y="team",
        orientation="h",
        color="team_conf",
        color_discrete_map=CONF_COLORS,
        template=template,
    )
    fig3.update_layout(
        **base_layout,
        height=520,
        legend_title_text="Conf",
        title=dict(text="Teams That Choke WORST (Avg Severity/Loss)", x=0.02, xanchor="left"),
    )
    fig3.update_yaxes(title="", automargin=True)
    fig3.update_xaxes(title="Avg True Choke Score per Loss")
    _lock_plotly_interactions(fig3)

    fig3_mobile = fig3.to_dict()
    fig3_mobile = go.Figure(fig3_mobile)
    fig3_mobile.update_layout(height=420, margin=dict(l=10, r=10, t=72, b=18))
    _lock_plotly_interactions(fig3_mobile)

    # -----------------------
    # CHART 4: Scatter
    # Desktop: markers + text
    # Mobile: markers only (no cramped labels)
    # -----------------------
    scatter_df = teams.sort_values("total_choke", ascending=False).head(24).copy()
    scatter_df["msize"] = 14 + 28 * clamp01(scatter_df["total_choke"])
    scatter_df = scatter_df.sort_values("total_choke", ascending=False).copy()

    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(
            x=scatter_df["choke_rate"],
            y=scatter_df["avg_choke_per_loss"],
            mode="markers+text",
            text=scatter_df["team"],
            textposition="top center",
            marker=dict(
                size=scatter_df["msize"],
                color=scatter_df["team_conf"].map(CONF_COLORS),
                line=dict(width=1, color="#0b0f14"),
                opacity=0.92,
            ),
            customdata=np.stack(
                [
                    scatter_df["logo"].astype(str),
                    scatter_df["total_choke"].round(2).astype(str),
                    scatter_df["playoff_losses"].round(0).astype(int).astype(str),
                    scatter_df["playoff_games"].round(0).astype(int).astype(str),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Choke Rate: %{x:.2f}<br>"
                "Avg Severity/Loss: %{y:.2f}<br>"
                "Total Choke: %{customdata[1]}<br>"
                "Playoff losses: %{customdata[2]} / games: %{customdata[3]}<br><br>"
                "<img src='%{customdata[0]}' style='height:52px;'>"
                "<extra></extra>"
            ),
        )
    )
    fig4.update_layout(
        **base_layout,
        height=650,
        showlegend=False,
        title=dict(text="Choke Most vs Choke Worst — (Top 24 Total Choke)", x=0.02, xanchor="left"),
    )
    fig4.update_xaxes(title="Choke Rate (Most)")
    fig4.update_yaxes(title="Avg Severity per Loss (Worst)")
    _lock_plotly_interactions(fig4)

    fig4_mobile = go.Figure()
    fig4_mobile.add_trace(
        go.Scatter(
            x=scatter_df["choke_rate"],
            y=scatter_df["avg_choke_per_loss"],
            mode="markers",
            marker=dict(
                size=10 + 14 * clamp01(scatter_df["total_choke"]),
                color=scatter_df["team_conf"].map(CONF_COLORS),
                line=dict(width=1, color="#0b0f14"),
                opacity=0.92,
            ),
            customdata=np.stack(
                [
                    scatter_df["team"].astype(str),
                    scatter_df["logo"].astype(str),
                    scatter_df["total_choke"].round(2).astype(str),
                    scatter_df["playoff_losses"].round(0).astype(int).astype(str),
                    scatter_df["playoff_games"].round(0).astype(int).astype(str),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Choke Rate: %{x:.2f}<br>"
                "Avg Severity/Loss: %{y:.2f}<br>"
                "Total Choke: %{customdata[2]}<br>"
                "Losses: %{customdata[3]} / games: %{customdata[4]}<br><br>"
                "<img src='%{customdata[1]}' style='height:52px;'>"
                "<extra></extra>"
            ),
        )
    )
    fig4_mobile.update_layout(
        **base_layout,
        height=460,
        showlegend=False,
        title=dict(text="Choke Most vs Choke Worst (Mobile View)", x=0.02, xanchor="left"),
        margin=dict(l=10, r=10, t=72, b=18),
    )
    fig4_mobile.update_xaxes(title="Choke Rate (Most)")
    fig4_mobile.update_yaxes(title="Avg Severity/Loss (Worst)")
    _lock_plotly_interactions(fig4_mobile)

    # -----------------------
    # Payloads for dropdown section + team deep dive
    # -----------------------
    deep_df = games.copy()
    for col in ["season", "week"]:
        if col not in deep_df.columns:
            deep_df[col] = np.nan

    preferred_cols = [
        "season", "week", "round", "opp", "points_for", "points_against",
        "max_lead", "true_choke_score", "round_unweighted_score"
    ]
    cols = [c for c in preferred_cols if c in deep_df.columns]

    deep_df = deep_df[["team"] + cols].copy()
    deep_df["team"] = deep_df["team"].astype(str).str.upper()

    deep_df["season"] = pd.to_numeric(deep_df.get("season", np.nan), errors="coerce")
    deep_df["week"] = pd.to_numeric(deep_df.get("week", np.nan), errors="coerce")
    deep_df["true_choke_score"] = pd.to_numeric(deep_df.get("true_choke_score", 0.0), errors="coerce").fillna(0.0)
    deep_df["round_unweighted_score"] = pd.to_numeric(deep_df.get("round_unweighted_score", 0.0), errors="coerce").fillna(0.0)
    deep_df["max_lead"] = pd.to_numeric(deep_df.get("max_lead", 0.0), errors="coerce").fillna(0.0)

    deep_df = deep_df.sort_values(["team", "true_choke_score"], ascending=[True, False]).copy()

    team_list = sorted([t for t in deep_df["team"].dropna().unique().tolist() if str(t).strip() != ""])
    team_payload = {t: deep_df[deep_df["team"] == t].to_dict(orient="records") for t in team_list}
    team_payload_json = json.dumps(team_payload)

    def _pack_top(df: pd.DataFrame, metric_col: str, n: int = 25):
        d = df.sort_values(metric_col, ascending=False).head(n).copy()
        if "label" not in d.columns:
            d["label"] = d.apply(build_label, axis=1)
        out = []
        for _, r in d.iterrows():
            out.append({
                "label": str(r.get("label", "")),
                "team": str(r.get("team", "")),
                "opp": str(r.get("opp", "")),
                "round": str(r.get("round", "")),
                "season": safe_int(r.get("season", "")),
                "max_lead": float(r.get("max_lead", 0.0)),
                "true_choke_score": float(r.get("true_choke_score", 0.0)),
                "round_unweighted_score": float(r.get("round_unweighted_score", 0.0)),
            })
        return out

    top_true_list = _pack_top(games, "true_choke_score", n=25)
    top_unw_list = _pack_top(games, "round_unweighted_score", n=25)

    rank_payload = {"true": top_true_list, "unweighted": top_unw_list}
    rank_payload_json = json.dumps(rank_payload)

    # -----------------------
    # KPI cards requested
    # 1) biggest choke game (true score)
    # 2) team highest overall total_choke
    # 3) team lowest overall total_choke
    # -----------------------
    build_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    biggest_game = games_top50_true.iloc[0]["label"] if len(games_top50_true) else "N/A"
    biggest_score = float(games_top50_true.iloc[0]["true_choke_score"]) if len(games_top50_true) else 0.0

    teams_sorted_total = teams.sort_values("total_choke", ascending=False).copy()
    highest_team = teams_sorted_total.iloc[0]["team"] if len(teams_sorted_total) else "N/A"
    highest_total = float(teams_sorted_total.iloc[0]["total_choke"]) if len(teams_sorted_total) else 0.0

    teams_sorted_low = teams.sort_values("total_choke", ascending=True).copy()
    lowest_team = teams_sorted_low.iloc[0]["team"] if len(teams_sorted_low) else "N/A"
    lowest_total = float(teams_sorted_low.iloc[0]["total_choke"]) if len(teams_sorted_low) else 0.0

    # -----------------------
    # HTML
    # -----------------------
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>NFL Playoff True Choke Index</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">

  <style>
    :root {{
      --bg: #0b0f14;
      --card: #0f1621;
      --border: #1f2a37;
      --text: #e6edf3;
      --muted: #9fb0c0;
      --chip: #111827;
      --accent: #7aa2ff;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 0; padding: 0;
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 18px 14px 56px 14px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: -0.3px;
    }}
    .sub {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
      margin: 10px 0 0 0;
    }}
    .hint {{
      color: var(--muted);
      font-size: 12px;
      margin: 8px 0 0 0;
    }}
    .explain {{
      margin-top: 10px;
      padding: 12px;
      background: rgba(15, 22, 33, 0.55);
      border: 1px solid rgba(31, 42, 55, 0.85);
      border-radius: 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .explain b {{ color: var(--text); }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 14px 0 16px 0;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .kpi .value {{
      font-size: 14px;
      font-weight: 800;
      line-height: 1.25;
    }}

    .grid2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 10px;
      margin: 12px 0;
      box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }}

    /* Plotly scroll behavior on touch */
    .js-plotly-plot, .plotly {{
      touch-action: pan-y !important;
    }}

    .section-title {{
      font-weight: 800;
      font-size: 16px;
      margin: 0 0 10px 0;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }}
    select {{
      background: var(--chip);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 14px;
      outline: none;
      width: min(420px, 100%);
    }}
    .pill {{
      padding: 8px 10px;
      background: var(--chip);
      border: 1px solid var(--border);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
    }}

    /* Desktop table */
    .table-wrap {{
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: auto;
      max-height: 520px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: #0e1520;
      color: var(--text);
      text-align: left;
      font-size: 12px;
      padding: 10px 10px;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}
    tbody td {{
      padding: 10px 10px;
      border-bottom: 1px solid rgba(31,42,55,0.55);
      color: var(--text);
      font-size: 13px;
      white-space: nowrap;
    }}
    tbody tr:hover td {{
      background: rgba(17,24,39,0.55);
    }}

    /* Mobile cards */
    .cards {{
      display: none;
      gap: 10px;
    }}
    .game-card {{
      background: #0c111a;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
    }}
    .game-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      font-weight: 800;
      font-size: 13px;
    }}
    .score-chip {{
      padding: 6px 10px;
      background: rgba(122,162,255,0.15);
      border: 1px solid rgba(122,162,255,0.35);
      border-radius: 999px;
      font-size: 12px;
      white-space: nowrap;
    }}
    .game-meta {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .meta-item b {{
      color: var(--text);
      font-weight: 700;
    }}

    /* MOBILE/DESKTOP versions */
    .desktop-only {{ display: block; }}
    .mobile-only {{ display: none; }}

    @media (max-width: 950px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
      .card {{ padding: 8px; }}
    }}

    @media (max-width: 700px) {{
      .desktop-only {{ display: none; }}
      .mobile-only {{ display: block; }}
      .table-wrap {{ display: none; }}
      .cards {{ display: grid; }}
    }}
  </style>
</head>

<body>
  <div class="wrap">
    <h1>NFL Playoff True Choke Index (2000–2025)</h1>

    <p class="sub">
      A “choke” here means a team had a meaningful lead (usually 10+ points) and still lost — then we score how bad it was based on
      <b>max lead</b>, <b>when the collapse happened</b>, and <b>special-teams meltdowns</b>.
      The charts below answer: <b>Which games were the biggest collapses?</b> and <b>Which franchises choke the most vs choke the worst?</b>
    </p>

    <div class="explain">
      <b>How to read this:</b><br>
      • <b>True Choke Score</b> = best “headline” metric (includes playoff-round importance).<br>
      • <b>Choke Most</b> = teams that repeatedly blow leads (rate).<br>
      • <b>Choke Worst</b> = teams whose losses are the most severe on average.<br>
      • Use the dropdown section to compare <b>with vs without round weighting</b>.
    </div>

    <p class="hint">Build: <b>{build_stamp}</b> (if this timestamp changes, your live site updated)</p>

    <div class="kpis">
      <div class="kpi">
        <div class="label">Biggest Choke Game (True Score)</div>
        <div class="value">{biggest_game}<br><span style="color:var(--muted);">Score: {biggest_score:.2f}</span></div>
      </div>
      <div class="kpi">
        <div class="label">Highest Total Franchise Choke</div>
        <div class="value">{highest_team}<br><span style="color:var(--muted);">Total: {highest_total:.2f}</span></div>
      </div>
      <div class="kpi">
        <div class="label">Lowest Total Franchise Choke</div>
        <div class="value">{lowest_team}<br><span style="color:var(--muted);">Total: {lowest_total:.2f}</span></div>
      </div>
    </div>

    <!-- =========================
         MOBILE VERSION (cleaner charts)
         ========================= -->
    <div class="mobile-only">
      <div class="card">
        {fig1_mobile.to_html(full_html=False, include_plotlyjs="cdn", config=config)}
      </div>

      <div class="card">
        {fig2_mobile.to_html(full_html=False, include_plotlyjs=False, config=config)}
      </div>

      <div class="card">
        {fig3_mobile.to_html(full_html=False, include_plotlyjs=False, config=config)}
      </div>

      <div class="card">
        {fig4_mobile.to_html(full_html=False, include_plotlyjs=False, config=config)}
      </div>
    </div>

    <!-- =========================
         DESKTOP VERSION (full layout)
         ========================= -->
    <div class="desktop-only">
      <div class="card">
        {fig1_desktop.to_html(full_html=False, include_plotlyjs="cdn", config=config)}
      </div>

      <div class="grid2">
        <div class="card">{fig2.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
        <div class="card">{fig3.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
      </div>

      <div class="card">{fig4.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
    </div>

    <!-- =========================
         Dropdown ranking compare
         ========================= -->
    <div class="card">
      <div class="section-title">Compare Rankings — With vs Without Playoff-Round Weight</div>
      <div class="controls">
        <select id="rankMode">
          <option value="true">True Choke Score (includes round importance)</option>
          <option value="unweighted">Round-Unweighted Score (ignores round importance)</option>
        </select>
        <div class="pill" id="rankHint">Showing: True Choke Score</div>
      </div>

      <div class="cards" id="rankCards"></div>

      <div class="table-wrap" style="max-height: 520px;">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Game</th>
              <th>True Score</th>
              <th>Unweighted</th>
              <th>Max Lead</th>
            </tr>
          </thead>
          <tbody id="rankTableBody"></tbody>
        </table>
      </div>

      <div class="hint" style="margin-top:10px;">
        “Unweighted” keeps your choke math but removes the playoff round multiplier — pure on-field collapse severity.
      </div>
    </div>

    <!-- =========================
         Team Deep Dive
         ========================= -->
    <div class="card">
      <div class="section-title">Team Deep Dive — All Chokes</div>
      <div class="controls">
        <select id="teamSelect"></select>
        <select id="teamMetric">
          <option value="true_choke_score">Sort by True Choke Score</option>
          <option value="round_unweighted_score">Sort by Round-Unweighted Score</option>
        </select>
        <div class="pill" id="rowCount">0 games</div>
        <div class="pill">Mobile: cards • Desktop: sortable table</div>
      </div>

      <div class="table-wrap">
        <table id="teamTable">
          <thead><tr id="theadRow"></tr></thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>

      <div class="cards" id="cards"></div>
    </div>

  </div>

<script>
  const TEAM_DATA = {team_payload_json};
  const RANK_DATA = {rank_payload_json};

  // ----------- Ranking section
  const rankMode = document.getElementById("rankMode");
  const rankHint = document.getElementById("rankHint");
  const rankTableBody = document.getElementById("rankTableBody");
  const rankCards = document.getElementById("rankCards");

  function fmt2(n) {{
    const x = Number(n);
    if (isNaN(x)) return "";
    return x.toFixed(2);
  }}

  function renderRank(mode) {{
    const rows = (RANK_DATA && RANK_DATA[mode]) ? RANK_DATA[mode] : [];
    rankHint.textContent = (mode === "true") ? "Showing: True Choke Score" : "Showing: Round-Unweighted Score";

    rankTableBody.innerHTML = "";
    for (let i = 0; i < rows.length; i++) {{
      const r = rows[i];
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + (i+1) + "</td>" +
        "<td>" + (r.label || "") + "</td>" +
        "<td>" + fmt2(r.true_choke_score) + "</td>" +
        "<td>" + fmt2(r.round_unweighted_score) + "</td>" +
        "<td>" + fmt2(r.max_lead) + "</td>";
      rankTableBody.appendChild(tr);
    }}

    rankCards.innerHTML = "";
    for (let i = 0; i < rows.length; i++) {{
      const r = rows[i];
      const div = document.createElement("div");
      div.className = "game-card";
      const headline = (r.season || "") + " " + (r.round || "") + " — " + (r.team || "") + " vs " + (r.opp || "");
      const chip = (mode === "true") ? ("True: " + fmt2(r.true_choke_score)) : ("Unw: " + fmt2(r.round_unweighted_score));
      div.innerHTML =
        "<div class='game-head'><div>" + (i+1) + ". " + headline + "</div><div class='score-chip'>" + chip + "</div></div>" +
        "<div class='game-meta'>" +
          "<div class='meta-item'><b>True</b><br>" + fmt2(r.true_choke_score) + "</div>" +
          "<div class='meta-item'><b>Unweighted</b><br>" + fmt2(r.round_unweighted_score) + "</div>" +
          "<div class='meta-item'><b>Max Lead</b><br>" + fmt2(r.max_lead) + "</div>" +
        "</div>";
      rankCards.appendChild(div);
    }}
  }}

  rankMode.addEventListener("change", (e) => {{
    renderRank(e.target.value);
  }});
  renderRank("true");


  // ----------- Team deep dive
  const teams = Object.keys(TEAM_DATA).sort();
  const defaultTeam = teams.includes("GB") ? "GB" : (teams[0] || "");

  const teamSelect = document.getElementById("teamSelect");
  const teamMetric = document.getElementById("teamMetric");
  const theadRow = document.getElementById("theadRow");
  const tbody = document.getElementById("tbody");
  const cards = document.getElementById("cards");
  const rowCount = document.getElementById("rowCount");

  const preferredCols = ["season","week","round","opp","points_for","points_against","max_lead","true_choke_score","round_unweighted_score"];

  let currentTeam = defaultTeam;
  let sortKey = "true_choke_score";
  let sortDir = "desc";

  function fmt(val, key) {{
    if (val === null || val === undefined) return "";
    if (key === "true_choke_score" || key === "round_unweighted_score") {{
      const n = Number(val);
      return isNaN(n) ? val : n.toFixed(2);
    }}
    if (key === "max_lead" || key === "points_for" || key === "points_against" || key === "week") {{
      const n = Number(val);
      return isNaN(n) ? val : n.toFixed(0);
    }}
    return val;
  }}

  function getCols(rows) {{
    if (!rows || rows.length === 0) return preferredCols;
    const keys = new Set();
    rows.forEach(r => Object.keys(r).forEach(k => keys.add(k)));
    return preferredCols.filter(c => keys.has(c));
  }}

  function sortRows(rows) {{
    const copy = rows.slice();
    copy.sort((a,b) => {{
      const av = a[sortKey];
      const bv = b[sortKey];

      const an = Number(av);
      const bn = Number(bv);
      const aNum = !isNaN(an);
      const bNum = !isNaN(bn);

      let res = 0;
      if (aNum && bNum) {{
        res = an - bn;
      }} else {{
        res = String(av ?? "").localeCompare(String(bv ?? ""));
      }}
      return sortDir === "asc" ? res : -res;
    }});
    return copy;
  }}

  function renderDesktopTable(sorted, cols) {{
    theadRow.innerHTML = "";
    cols.forEach(col => {{
      const th = document.createElement("th");
      th.textContent = col.replaceAll("_"," ").toUpperCase();
      th.addEventListener("click", () => {{
        if (sortKey === col) {{
          sortDir = (sortDir === "asc") ? "desc" : "asc";
        }} else {{
          sortKey = col;
          sortDir = (col.includes("score")) ? "desc" : "asc";
        }}
        renderTeam(currentTeam);
      }});
      theadRow.appendChild(th);
    }});

    tbody.innerHTML = "";
    sorted.forEach(r => {{
      const tr = document.createElement("tr");
      cols.forEach(col => {{
        const td = document.createElement("td");
        td.textContent = fmt(r[col], col);
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});
  }}

  function renderMobileCards(sorted) {{
    cards.innerHTML = "";
    sorted.forEach(r => {{
      const season = fmt(r.season, "season");
      const rnd = r.round || "";
      const opp = r.opp || "";
      const pf = fmt(r.points_for, "points_for");
      const pa = fmt(r.points_against, "points_against");
      const lead = fmt(r.max_lead, "max_lead");
      const tscore = fmt(r.true_choke_score, "true_choke_score");
      const uscore = fmt(r.round_unweighted_score, "round_unweighted_score");

      const div = document.createElement("div");
      div.className = "game-card";
      div.innerHTML =
        "<div class='game-head'>" +
          "<div>" + season + " " + rnd + " vs " + opp + "</div>" +
          "<div class='score-chip'>True: " + tscore + "</div>" +
        "</div>" +
        "<div class='game-meta'>" +
          "<div class='meta-item'><b>Final</b><br>" + pf + "-" + pa + "</div>" +
          "<div class='meta-item'><b>Max Lead</b><br>" + lead + "</div>" +
          "<div class='meta-item'><b>Unweighted</b><br>" + uscore + "</div>" +
        "</div>";
      cards.appendChild(div);
    }});
  }}

  function renderTeam(team) {{
    currentTeam = team;
    const rows = TEAM_DATA[team] || [];
    const cols = getCols(rows);

    sortKey = teamMetric.value;
    sortDir = "desc";

    const sorted = sortRows(rows);
    rowCount.textContent = sorted.length + " games";

    renderDesktopTable(sorted, cols);
    renderMobileCards(sorted);
  }}

  teams.forEach(t => {{
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    teamSelect.appendChild(opt);
  }});
  teamSelect.value = defaultTeam;

  teamSelect.addEventListener("change", (e) => {{
    renderTeam(e.target.value);
  }});
  teamMetric.addEventListener("change", () => {{
    renderTeam(currentTeam);
  }});

  renderTeam(defaultTeam);
</script>

</body>
</html>
"""

    with open(INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Dashboard written to:")
    print("   ", INDEX_HTML)
    print("   ", DASHBOARD_HTML)
    print("Build stamp:", build_stamp)


if __name__ == "__main__":
    main()
