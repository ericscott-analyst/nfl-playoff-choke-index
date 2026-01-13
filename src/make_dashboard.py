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


def main():
    require_file(TOP_GAMES_CSV)
    require_file(TEAM_RANK_CSV)

    games = pd.read_csv(TOP_GAMES_CSV)
    teams = pd.read_csv(TEAM_RANK_CSV)

    # ---------- Clean/standardize expected columns ----------
    # top games expected columns (based on your generator)
    for col in ["true_choke_score", "max_lead", "importance_weight", "late_weight", "points_for", "points_against"]:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce").fillna(0)

    for col in ["season", "week"]:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce")

    # Create a readable label for charts
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

    games = games.sort_values("true_choke_score", ascending=False).head(50).copy()

    # Team rankings cleanup
    for col in ["total_choke", "avg_choke_per_loss", "choke_rate", "playoff_losses", "playoff_games", "true_chokes_10"]:
        if col in teams.columns:
            teams[col] = pd.to_numeric(teams[col], errors="coerce").fillna(0)

    # Some earlier versions name columns slightly differently
    # Normalize if needed
    if "total_choke" not in teams.columns and "total_true_choke" in teams.columns:
        teams["total_choke"] = teams["total_true_choke"]
    if "avg_choke_per_loss" not in teams.columns and "avg_true_choke_per_loss" in teams.columns:
        teams["avg_choke_per_loss"] = teams["avg_true_choke_per_loss"]
    if "true_chokes_10" not in teams.columns and "true_chokes_10" not in teams.columns:
        pass

    # ---------- Plotly theme ----------
    template = "plotly_dark"

    # ---------- Chart 1: Top 25 choke games ----------
    top25 = games.head(25).iloc[::-1].copy()
    fig1 = px.bar(
        top25,
        x="true_choke_score",
        y="label",
        orientation="h",
        title="Top 25 Biggest Playoff Chokes (True Choke Score)",
        hover_data={
            "true_choke_score": ":.2f",
            "max_lead": True,
            "importance_weight": ":.2f",
            "late_weight": ":.2f",
        },
        template=template
    )
    fig1.update_layout(height=850, margin=dict(l=20, r=20, t=60, b=20))
    fig1.update_yaxes(title="Game", automargin=True)
    fig1.update_xaxes(title="True Choke Score")

    # ---------- Chart 2: Top 20 teams by total choke severity ----------
    teams_sorted = teams.sort_values("total_choke", ascending=False).head(20).copy()
    fig2 = px.bar(
        teams_sorted.iloc[::-1],
        x="total_choke",
        y="team",
        orientation="h",
        title="Top 20 Teams by Total Playoff Choke Score (2000–2025)",
        hover_data={
            "total_choke": ":.2f",
            "avg_choke_per_loss": ":.2f",
            "playoff_losses": True,
            "playoff_games": True
        },
        template=template
    )
    fig2.update_layout(height=650, margin=dict(l=20, r=20, t=60, b=20))
    fig2.update_yaxes(title="Team", automargin=True)
    fig2.update_xaxes(title="Total Choke Score")

    # ---------- Chart 3: Choke Most vs Choke Worst (scatter) ----------
    # "Most" proxy = choke_rate, "Worst" proxy = avg_choke_per_loss
    if "choke_rate" in teams.columns and "avg_choke_per_loss" in teams.columns:
        fig3 = px.scatter(
            teams,
            x="choke_rate",
            y="avg_choke_per_loss",
            size="total_choke" if "total_choke" in teams.columns else None,
            hover_name="team",
            title="Choke Most vs Choke Worst (size = total choke score)",
            template=template
        )
        fig3.update_layout(height=650, margin=dict(l=20, r=20, t=60, b=20))
        fig3.update_xaxes(title="Choke Rate (10+ lead blown per playoff loss)")
        fig3.update_yaxes(title="Avg Choke Score per Loss")
    else:
        fig3 = go.Figure()
        fig3.update_layout(template=template, title="Choke Most vs Choke Worst (missing columns)")

    # ---------- Simple team dropdown table (top 3 chokes) ----------
    # We'll compute top 3 for each team from the games file.
    top_by_team = games.sort_values("true_choke_score", ascending=False).groupby("team").head(3).copy()
    top_by_team["display"] = top_by_team["label"] + " | Score " + top_by_team["true_choke_score"].round(2).astype(str)

    team_list = sorted(top_by_team["team"].dropna().unique().tolist())
    if len(team_list) == 0:
        team_list = ["(no teams found)"]

    # Create dropdown traces
    table_traces = []
    for t in team_list:
        sub = top_by_team[top_by_team["team"] == t].copy()
        rows = sub["display"].tolist()
        if len(rows) == 0:
            rows = ["No chokes found for this team in top 50."]
        table_traces.append(rows)

    # Initial table
    initial_rows = table_traces[0] if table_traces else ["No data."]
    fig4 = go.Figure()
    fig4.add_trace(go.Table(
        header=dict(values=["Top 3 Chokes (from Top 50 list)"], fill_color="black"),
        cells=dict(values=[initial_rows])
    ))

    # Dropdown buttons
    buttons = []
    for i, t in enumerate(team_list):
        buttons.append(dict(
            label=t,
            method="restyle",
            args=[{"cells.values": [[table_traces[i]]]}]
        ))

    fig4.update_layout(
        template=template,
        title="Team Deep Dive — Top 3 Chokes (Dropdown)",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.01,
            y=1.12,
            xanchor="left",
            yanchor="top"
        )]
    )

    # ---------- Build final HTML ----------
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>NFL Playoff True Choke Index Dashboard</title>
  <style>
    body {{
      background: #0b0f14;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 0;
      padding: 0;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 28px;
    }}
    p {{
      margin: 6px 0 18px 0;
      color: #9fb0c0;
      line-height: 1.4;
    }}
    .card {{
      background: #0f1621;
      border: 1px solid #1f2a37;
      border-radius: 14px;
      padding: 14px;
      margin: 14px 0;
    }}
    .small {{
      font-size: 13px;
      color: #9fb0c0;
    }}
    a {{
      color: #7aa2ff;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>NFL Playoff True Choke Index (2000–2025)</h1>
    <p>
      Interactive dashboard generated from play-by-play data. Scoring includes game importance, late-game collapse weighting, and special teams disasters.
    </p>
    <p class="small">
      Tip: Hover points/bars for details. Zoom and pan in charts. Use the dropdown for team deep dives.
    </p>

    <div class="card">{fig1.to_html(full_html=False, include_plotlyjs="cdn")}</div>
    <div class="card">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="card">{fig3.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="card">{fig4.to_html(full_html=False, include_plotlyjs=False)}</div>

    <p class="small">
      Generated by <b>make_dashboard.py</b>. Source code and methodology available in the repo README.
    </p>
  </div>
</body>
</html>
"""

    with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    # Simple docs index page
    index = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>NFL Playoff True Choke Index</title>
  <style>
    body { background:#0b0f14; color:#e6edf3; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; padding:24px; }
    a { color:#7aa2ff; }
    .card { background:#0f1621; border:1px solid #1f2a37; border-radius:14px; padding:16px; max-width:900px; }
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
    print("\nOpen this file in your browser:")
    print("   ", INDEX_HTML)


if __name__ == "__main__":
    main()
