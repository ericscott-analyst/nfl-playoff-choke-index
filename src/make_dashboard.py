import os
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

DASHBOARD_HTML = os.path.join(DOCS_DIR, "dashboard.html")
INDEX_HTML = os.path.join(DOCS_DIR, "index.html")


def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n\nRun your build script first to generate outputs/*.csv"
        )


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


def build_team_color_map(teams):
    """
    Deterministic team -> color mapping (no web lookup needed).
    Uses a large qualitative palette and assigns consistently.
    """
    # Big combined palette
    palette = (
        px.colors.qualitative.Bold
        + px.colors.qualitative.D3
        + px.colors.qualitative.G10
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
    )
    uniq = sorted([t for t in pd.Series(teams).dropna().unique().tolist() if str(t).strip() != ""])
    color_map = {}
    for i, t in enumerate(uniq):
        color_map[t] = palette[i % len(palette)]
    return color_map


def main():
    require_file(TOP_GAMES_CSV)
    require_file(TEAM_RANK_CSV)

    games = pd.read_csv(TOP_GAMES_CSV)
    teams = pd.read_csv(TEAM_RANK_CSV)

    # ---------- Normalize columns ----------
    # Games
    for col in ["true_choke_score", "max_lead", "importance_weight", "late_weight", "points_for", "points_against"]:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce").fillna(0)

    for col in ["season", "week"]:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce")

    # Create readable label
    def build_label(r):
        season = safe_int(r.get("season", ""))
        rnd = r.get("round", "")
        team = r.get("team", "")
        opp = r.get("opp", "")
        pf = safe_int(r.get("points_for", ""))
        pa = safe_int(r.get("points_against", ""))
        lead = safe_int(r.get("max_lead", ""))
        return f"{season} {rnd}: {team} vs {opp} ({pf}-{pa}) | Lead {lead}"

    if "label" not in games.columns:
        games["label"] = games.apply(build_label, axis=1)

    # Teams table
    for col in ["total_choke", "avg_choke_per_loss", "choke_rate", "playoff_losses", "playoff_games", "true_chokes_10"]:
        if col in teams.columns:
            teams[col] = pd.to_numeric(teams[col], errors="coerce").fillna(0)

    # Normalize alternate column names if needed
    if "total_choke" not in teams.columns and "total_true_choke" in teams.columns:
        teams["total_choke"] = teams["total_true_choke"]
    if "avg_choke_per_loss" not in teams.columns and "avg_true_choke_per_loss" in teams.columns:
        teams["avg_choke_per_loss"] = teams["avg_true_choke_per_loss"]

    # ---------- Sort ----------
    games = games.sort_values("true_choke_score", ascending=False).head(50).copy()
    teams = teams.copy()

    # ---------- Team colors ----------
    team_color_map = build_team_color_map(pd.concat([games["team"], teams.get("team", pd.Series(dtype=str))], ignore_index=True))

    # ---------- Plotly theme ----------
    template = "plotly_dark"
    base_layout = dict(
        template=template,
        font=dict(size=14),
        margin=dict(l=30, r=30, t=70, b=30),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        title=dict(x=0.02, xanchor="left")
    )

    # ---------- Chart 1: Top 25 choke games (colored by team) ----------
    top25 = games.head(25).iloc[::-1].copy()

    fig1 = px.bar(
        top25,
        x="true_choke_score",
        y="label",
        orientation="h",
        color="team",
        color_discrete_map=team_color_map,
        title="Top 25 Biggest Playoff Chokes (True Choke Score)",
        hover_data={
            "team": True,
            "opp": True,
            "true_choke_score": ":.2f",
            "max_lead": True,
            "importance_weight": ":.2f",
            "late_weight": ":.2f",
        },
        template=template
    )
    fig1.update_layout(**base_layout, height=980, legend_title_text="Team")
    fig1.update_yaxes(title="", automargin=True)
    fig1.update_xaxes(title="True Choke Score")

    # ---------- Chart 2: Top 20 teams by total choke (colored by team) ----------
    teams_sorted = teams.sort_values("total_choke", ascending=False).head(20).copy()

    fig2 = px.bar(
        teams_sorted.iloc[::-1],
        x="total_choke",
        y="team",
        orientation="h",
        color="team",
        color_discrete_map=team_color_map,
        title="Top 20 Teams by Total Playoff Choke Score (2000–2025)",
        hover_data={
            "total_choke": ":.2f",
            "avg_choke_per_loss": ":.2f",
            "playoff_losses": True,
            "playoff_games": True
        },
        template=template
    )
    fig2.update_layout(**base_layout, height=760, showlegend=False)
    fig2.update_yaxes(title="", automargin=True)
    fig2.update_xaxes(title="Total Choke Score")

    # ---------- Chart 3: Scatter (color by team for differentiating) ----------
    if "choke_rate" in teams.columns and "avg_choke_per_loss" in teams.columns:
        # Keep chart readable: show top 32 by total choke; otherwise too many points/colors
        scatter_df = teams.sort_values("total_choke", ascending=False).head(32).copy()

        fig3 = px.scatter(
            scatter_df,
            x="choke_rate",
            y="avg_choke_per_loss",
            size="total_choke" if "total_choke" in scatter_df.columns else None,
            color="team",
            color_discrete_map=team_color_map,
            hover_name="team",
            title="Choke Most vs Choke Worst (Top 32 teams by total choke)",
            template=template
        )
        fig3.update_layout(**base_layout, height=720, legend_title_text="Team")
        fig3.update_xaxes(title="Choke Rate (10+ leads blown per playoff loss)")
        fig3.update_yaxes(title="Avg Choke Score per Loss")
    else:
        fig3 = go.Figure()
        fig3.update_layout(**base_layout, title="Choke Most vs Choke Worst (missing columns)", height=520)

    # ---------- Team dropdown table (Top 3 chokes) ----------
    top_by_team = games.sort_values("true_choke_score", ascending=False).groupby("team").head(3).copy()
    top_by_team["display"] = top_by_team["label"] + " | Score " + top_by_team["true_choke_score"].round(2).astype(str)

    team_list = sorted([t for t in top_by_team["team"].dropna().unique().tolist() if str(t).strip() != ""])
    if len(team_list) == 0:
        team_list = ["(no teams found)"]

    table_traces = []
    for t in team_list:
        sub = top_by_team[top_by_team["team"] == t].copy()
        rows = sub["display"].tolist()
        if len(rows) == 0:
            rows = ["No chokes found for this team in the Top 50 list."]
        table_traces.append(rows)

    fig4 = go.Figure()
    fig4.add_trace(go.Table(
        header=dict(
            values=["Team Deep Dive — Top 3 Chokes (from Top 50 list)"],
            fill_color="#0f1621",
            font=dict(color="#e6edf3", size=14),
            align="left"
        ),
        cells=dict(
            values=[table_traces[0] if table_traces else ["No data."]],
            fill_color="#0b0f14",
            font=dict(color="#e6edf3", size=13),
            align="left",
            height=28
        )
    ))

    buttons = []
    for i, t in enumerate(team_list):
        buttons.append(dict(
            label=t,
            method="restyle",
            args=[{"cells.values": [[table_traces[i]]]}]
        ))

    fig4.update_layout(
        **base_layout,
        title="Team Deep Dive (Dropdown)",
        height=560,
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.02,
            y=1.13,
            xanchor="left",
            yanchor="top",
            bgcolor="#0f1621",
            bordercolor="#1f2a37"
        )]
    )

    # ---------- KPI cards (sports analytics report feel) ----------
    total_games = int(len(pd.read_csv(os.path.join(OUTPUTS_DIR, "all_team_games_playoffs.csv")))) if os.path.exists(os.path.join(OUTPUTS_DIR, "all_team_games_playoffs.csv")) else int(len(games))
    total_chokes = int((games["true_choke_score"] > 0).sum())
    biggest_game = games.iloc[0]["label"] if len(games) else "N/A"
    biggest_score = float(games.iloc[0]["true_choke_score"]) if len(games) else 0.0
    top_team = teams.sort_values("total_choke", ascending=False).iloc[0]["team"] if len(teams) else "N/A"

    # ---------- Build HTML (narrower + taller report layout) ----------
    # Make charts responsive and not "super wide/short"
    config = {"responsive": True, "displaylogo": False}

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>NFL Playoff True Choke Index Dashboard</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --card: #0f1621;
      --border: #1f2a37;
      --text: #e6edf3;
      --muted: #9fb0c0;
      --accent: #7aa2ff;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 0;
      padding: 0;
    }}
    .wrap {{
      max-width: 1080px;   /* narrower = more report-like */
      margin: 0 auto;
      padding: 22px 16px 40px 16px;
    }}
    header {{
      padding: 6px 0 10px 0;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 28px;
      letter-spacing: 0.2px;
    }}
    p {{
      margin: 6px 0 10px 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0 16px 0;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 12px;
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .kpi .value {{
      font-size: 18px;
      font-weight: 700;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px;
      margin: 12px 0;
      box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }}
    .small {{
      font-size: 12px;
      color: var(--muted);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    @media (max-width: 880px) {{
      .kpis {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>NFL Playoff True Choke Index (2000–2025)</h1>
      <p>
        Interactive sports-analytics report built from play-by-play data. Scoring blends game importance, late-game collapse weighting, and special teams disasters.
      </p>
      <p class="small">Tip: hover for details · zoom/pan charts · use the dropdown for team deep dives.</p>
    </header>

    <section class="kpis">
      <div class="kpi">
        <div class="label">Total playoff team-games (dataset)</div>
        <div class="value">{total_games:,}</div>
      </div>
      <div class="kpi">
        <div class="label">Choke games in Top 50 list</div>
        <div class="value">{total_chokes:,}</div>
      </div>
      <div class="kpi">
        <div class="label">#1 choke game</div>
        <div class="value" style="font-size:14px; font-weight:600">{biggest_game}<br><span class="small">Score: {biggest_score:.2f}</span></div>
      </div>
    </section>

    <div class="card">{fig1.to_html(full_html=False, include_plotlyjs="cdn", config=config)}</div>
    <div class="card">{fig2.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
    <div class="card">{fig3.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
    <div class="card">{fig4.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>

    <p class="small">
      Generated by <b>make_dashboard.py</b>. Repo: <a href="https://github.com/">GitHub</a>
      &nbsp;·&nbsp; Top team by total choke score: <b>{top_team}</b>
    </p>
  </div>
</body>
</html>
"""

    with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    index = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>NFL Playoff True Choke Index</title>
  <style>
    body { background:#0b0f14; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; padding:24px; }
    a { color:#7aa2ff; }
    .card { background:#0f1621; border:1px solid #1f2a37; border-radius:16px; padding:16px; max-width:900px; box-shadow:0 10px 25px rgba(0,0,0,0.25); }
    p { color:#9fb0c0; }
  </style>
</head>
<body>
  <div class="card">
    <h2>NFL Playoff True Choke Index (2000–2025)</h2>
    <p>Open the interactive dashboard:</p>
    <p><a href="./dashboard.html">dashboard.html</a></p>
  </div>
</body>
</html>
"""
    with open(INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(index)

    print("✅ Dashboard generated:")
    print("   ", DASHBOARD_HTML)
    print("✅ Docs index generated:")
    print("   ", INDEX_HTML)
    print("\nOpen in your browser:")
    print("   ", INDEX_HTML)


if __name__ == "__main__":
    main()
