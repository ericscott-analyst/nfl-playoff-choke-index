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

INDEX_HTML = os.path.join(DOCS_DIR, "index.html")
DASHBOARD_HTML = os.path.join(DOCS_DIR, "dashboard.html")


def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n\nRun your build script first to generate outputs/*.csv"
        )


def to_num(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


def build_label(row):
    season = safe_int(row.get("season", ""))
    rnd = row.get("round", "")
    team = row.get("team", "")
    opp = row.get("opp", "")
    pf = safe_int(row.get("points_for", ""))
    pa = safe_int(row.get("points_against", ""))
    lead = safe_int(row.get("max_lead", ""))
    return f"{season} {rnd}: {team} vs {opp} ({pf}-{pa}) | Lead {lead}"


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


def clamp01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def try_load_team_meta():
    """
    Best-effort team metadata (logos + conf). If it fails (no internet), we just skip logos.
    """
    url = "https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/teams_colors_logos.csv"
    try:
        meta = pd.read_csv(url)
        # Standardize team column name to "team"
        if "team_abbr" in meta.columns:
            meta = meta.rename(columns={"team_abbr": "team"})
        meta["team"] = meta["team"].astype(str).str.upper()

        # Prefer ESPN logo, fallback to wikipedia if needed
        if "team_logo_espn" not in meta.columns and "team_logo_wikipedia" in meta.columns:
            meta["team_logo_espn"] = meta["team_logo_wikipedia"]

        keep = [c for c in ["team", "team_conf", "team_logo_espn"] if c in meta.columns]
        meta = meta[keep].copy()
        return meta
    except Exception:
        return None


def main():
    require_file(TOP_GAMES_CSV)
    require_file(TEAM_RANK_CSV)

    games = pd.read_csv(TOP_GAMES_CSV)
    teams = pd.read_csv(TEAM_RANK_CSV)

    # ---- Normalize games ----
    # Ensure expected numeric cols exist
    for col in ["true_choke_score", "max_lead", "importance_weight", "late_weight", "points_for", "points_against"]:
        games = to_num(games, col, 0.0)

    # Ensure expected string cols exist
    for col in ["team", "opp", "round"]:
        if col not in games.columns:
            games[col] = ""

    # Label + round bucket
    if "label" not in games.columns:
        games["label"] = games.apply(build_label, axis=1)

    games["round_bucket"] = games["round"].apply(round_bucket)

    games["team"] = games["team"].astype(str).str.upper()
    games["opp"] = games["opp"].astype(str).str.upper()

    games = games.sort_values("true_choke_score", ascending=False).head(50).copy()

    # ---- Normalize teams ----
    # Handle alternate column names
    if "total_choke" not in teams.columns and "total_true_choke" in teams.columns:
        teams["total_choke"] = teams["total_true_choke"]
    if "avg_choke_per_loss" not in teams.columns and "avg_true_choke_per_loss" in teams.columns:
        teams["avg_choke_per_loss"] = teams["avg_true_choke_per_loss"]

    for col in ["team", "total_choke", "avg_choke_per_loss", "choke_rate", "playoff_losses", "playoff_games"]:
        if col == "team":
            if col not in teams.columns:
                teams[col] = ""
            teams[col] = teams[col].astype(str).str.upper()
        else:
            teams = to_num(teams, col, 0.0)

    # ---- Team meta (logos + conference) ----
    meta = try_load_team_meta()
    logo_map = {}
    conf_map = {}

    if meta is not None:
        if "team_logo_espn" in meta.columns:
            logo_map = dict(zip(meta["team"], meta["team_logo_espn"]))
        if "team_conf" in meta.columns:
            conf_map = dict(zip(meta["team"], meta["team_conf"]))

    teams["team_conf"] = teams["team"].map(conf_map).fillna("Unknown")
    teams["logo"] = teams["team"].map(logo_map).fillna("")

    # ---- Styling ----
    template = "plotly_dark"
    base_layout = dict(
        template=template,
        font=dict(size=14),
        margin=dict(l=26, r=26, t=68, b=26),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
    )
    config = {"responsive": True, "displaylogo": False}

    ROUND_COLORS = {
        "Wild Card": "#60a5fa",
        "Divisional": "#34d399",
        "Conference": "#fbbf24",
        "Super Bowl": "#f87171",
        "Unknown": "#9ca3af"
    }
    CONF_COLORS = {"AFC": "#60a5fa", "NFC": "#34d399", "Unknown": "#9ca3af"}

    # =========================
    # CHART 1 — Top 25 games
    # =========================
    top25 = games.head(25).iloc[::-1].copy()
    fig1 = px.bar(
        top25,
        x="true_choke_score",
        y="label",
        orientation="h",
        color="round_bucket",
        color_discrete_map=ROUND_COLORS,
        title="Top 25 Biggest Playoff Chokes (True Choke Score)",
        hover_data={
            "team": True,
            "opp": True,
            "round_bucket": True,
            "true_choke_score": ":.2f",
            "max_lead": True,
            "importance_weight": ":.2f",
            "late_weight": ":.2f",
        },
        template=template
    )
    fig1.update_layout(**base_layout, height=920, legend_title_text="Round", title=dict(x=0.02, xanchor="left"))
    fig1.update_yaxes(title="", automargin=True)
    fig1.update_xaxes(title="True Choke Score")

    # =================================
    # CHART 2 — Teams that choke MOST
    # =================================
    most_df = teams.sort_values("choke_rate", ascending=False).head(16).copy()
    fig2 = px.bar(
        most_df.iloc[::-1],
        x="choke_rate",
        y="team",
        orientation="h",
        color="team_conf",
        color_discrete_map=CONF_COLORS,
        title="Teams That Choke MOST (Rate)",
        hover_data={
            "choke_rate": ":.2f",
            "playoff_losses": True,
            "playoff_games": True,
            "total_choke": ":.2f",
            "avg_choke_per_loss": ":.2f"
        },
        template=template
    )
    fig2.update_layout(**base_layout, height=520, legend_title_text="Conf", title=dict(x=0.02, xanchor="left"))
    fig2.update_yaxes(title="", automargin=True)
    fig2.update_xaxes(title="Choke Rate (10+ leads blown per playoff loss)")

    # ==================================
    # CHART 3 — Teams that choke WORST
    # ==================================
    worst_df = teams.sort_values("avg_choke_per_loss", ascending=False).head(16).copy()
    fig3 = px.bar(
        worst_df.iloc[::-1],
        x="avg_choke_per_loss",
        y="team",
        orientation="h",
        color="team_conf",
        color_discrete_map=CONF_COLORS,
        title="Teams That Choke WORST (Avg Severity per Loss)",
        hover_data={
            "avg_choke_per_loss": ":.2f",
            "playoff_losses": True,
            "total_choke": ":.2f",
            "choke_rate": ":.2f"
        },
        template=template
    )
    fig3.update_layout(**base_layout, height=520, legend_title_text="Conf", title=dict(x=0.02, xanchor="left"))
    fig3.update_yaxes(title="", automargin=True)
    fig3.update_xaxes(title="Avg True Choke Score per Loss")

    # ============================================
    # CHART 4 — Choke Most vs Choke Worst (logos)
    # ============================================
    scatter_df = teams.sort_values("total_choke", ascending=False).head(24).copy()

    # Marker sizes based on total choke
    scatter_df["msize"] = 14 + 28 * clamp01(scatter_df["total_choke"])

    # We sort by total choke so it feels “most -> least” in the hover ordering
    scatter_df = scatter_df.sort_values("total_choke", ascending=False).copy()

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=scatter_df["choke_rate"],
        y=scatter_df["avg_choke_per_loss"],
        mode="markers+text",
        text=scatter_df["team"],
        textposition="top center",
        marker=dict(
            size=scatter_df["msize"],
            color=scatter_df["team_conf"].map(CONF_COLORS),
            line=dict(width=1, color="#0b0f14"),
            opacity=0.92
        ),
        customdata=np.stack([
            scatter_df["logo"].astype(str),
            scatter_df["total_choke"].round(2).astype(str),
            scatter_df["playoff_losses"].round(0).astype(int).astype(str),
            scatter_df["playoff_games"].round(0).astype(int).astype(str),
        ], axis=1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Choke Rate: %{x:.2f}<br>"
            "Avg Severity/Loss: %{y:.2f}<br>"
            "Total Choke: %{customdata[1]}<br>"
            "Playoff losses: %{customdata[2]} / games: %{customdata[3]}<br>"
            "<br>"
            "<img src='%{customdata[0]}' style='height:52px;'>"
            "<extra></extra>"
        )
    ))
    fig4.update_layout(
        **base_layout,
        height=700,
        title="Choke Most vs Choke Worst (Top 24 by Total Choke) — Logos in Hover",
        showlegend=False
    )
    fig4.update_xaxes(title="Choke Rate (Most)")
    fig4.update_yaxes(title="Avg Choke Severity per Loss (Worst)")

    # ============================================
    # TEAM DEEP DIVE — Dropdown + clean chart/table
    # ============================================
    top_by_team = games.sort_values("true_choke_score", ascending=False).copy()
    team_list = sorted([t for t in top_by_team["team"].dropna().unique().tolist() if str(t).strip() != ""])
    if not team_list:
        team_list = ["N/A"]

    # Prepare per-team bar data + table rows
    bar_data = {}
    table_data = {}
    for t in team_list:
        sub = top_by_team[top_by_team["team"] == t].head(8).copy()
        # nicer order for horizontal bars
        sub = sub.sort_values("true_choke_score", ascending=True)
        bar_data[t] = (sub["true_choke_score"].tolist(), sub["label"].tolist())

        t3 = top_by_team[top_by_team["team"] == t].head(3).copy()
        if t3.empty:
            table_data[t] = ["No games for this team in the Top 50 list."]
        else:
            table_data[t] = (t3["label"] + " | Score " + t3["true_choke_score"].round(2).astype(str)).tolist()

    # Bar chart
    first_team = team_list[0]
    x0, y0 = bar_data[first_team]

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=x0,
        y=y0,
        orientation="h",
        marker=dict(color="#60a5fa"),
        hovertemplate="Score: %{x:.2f}<br>%{y}<extra></extra>",
        name="Top Chokes"
    ))
    fig5.update_layout(
        **base_layout,
        height=520,
        title=f"Team Deep Dive — {first_team}: Top Chokes (Top 50 list)",
    )
    fig5.update_xaxes(title="True Choke Score")
    fig5.update_yaxes(title="", automargin=True)

    bar_buttons = []
    for t in team_list:
        xs, ys = bar_data[t]
        bar_buttons.append(dict(
            label=t,
            method="update",
            args=[
                {"x": [xs], "y": [ys]},
                {"title": {"text": f"Team Deep Dive — {t}: Top Chokes (Top 50 list)", "x": 0.02, "xanchor": "left"}}
            ]
        ))
    fig5.update_layout(
        updatemenus=[dict(
            buttons=bar_buttons,
            direction="down",
            x=0.02, y=1.17,
            xanchor="left", yanchor="top",
            bgcolor="#0f1621",
            bordercolor="#1f2a37"
        )]
    )

    # Table chart
    fig6 = go.Figure()
    fig6.add_trace(go.Table(
        header=dict(
            values=["Team Deep Dive — Top 3 Chokes (Summary)"],
            fill_color="#0f1621",
            font=dict(color="#e6edf3", size=14),
            align="left"
        ),
        cells=dict(
            values=[table_data[first_team]],
            fill_color="#0b0f14",
            font=dict(color="#e6edf3", size=13),
            align="left",
            height=28
        )
    ))
    fig6.update_layout(
        **base_layout,
        height=360,
        title=f"Team Deep Dive — {first_team}: Top 3 (Top 50 list)",
    )

    table_buttons = []
    for t in team_list:
        table_buttons.append(dict(
            label=t,
            method="restyle",
            args=[{"cells.values": [[table_data[t]]]}]
        ))
    fig6.update_layout(
        updatemenus=[dict(
            buttons=table_buttons,
            direction="down",
            x=0.02, y=1.17,
            xanchor="left", yanchor="top",
            bgcolor="#0f1621",
            bordercolor="#1f2a37"
        )]
    )

    # ---- KPI cards ----
    top_team_total = teams.sort_values("total_choke", ascending=False).head(1)
    top_team_name = top_team_total["team"].iloc[0] if len(top_team_total) else "N/A"
    top_team_score = float(top_team_total["total_choke"].iloc[0]) if len(top_team_total) else 0.0

    biggest_game = games.iloc[0]["label"] if len(games) else "N/A"
    biggest_score = float(games.iloc[0]["true_choke_score"]) if len(games) else 0.0

    # ---- Modern HTML: index.html IS the dashboard (no extra click) ----
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
      --accent: #7aa2ff;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 0;
      padding: 0;
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 22px 16px 48px 16px;
    }}
    .hero {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 12px;
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      letter-spacing: -0.3px;
    }}
    .sub {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
      margin: 0;
    }}
    .hint {{
      color: var(--muted);
      font-size: 12px;
      margin: 0;
    }}
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
      padding: 12px 12px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .kpi .value {{
      font-size: 18px;
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
      padding: 12px;
      margin: 12px 0;
      box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }}
    .footer {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 950px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>NFL Playoff True Choke Index (2000–2025)</h1>
      <p class="sub">
        Portfolio-ready sports analytics report built from play-by-play data.
        Clean color choices: <b>Round</b> for games and <b>Conference</b> for team charts to avoid rainbow confusion.
        Team logos appear in the scatter hover cards.
      </p>
      <p class="hint">Tip: Hover for details • Zoom/Pan charts • Use dropdowns in Team Deep Dive</p>
    </div>

    <div class="kpis">
      <div class="kpi">
        <div class="label">#1 Choke Game</div>
        <div class="value" style="font-size:14px;font-weight:700">{biggest_game}<br><span class="hint">Score: {biggest_score:.2f}</span></div>
      </div>
      <div class="kpi">
        <div class="label">Top Franchise (Total Choke)</div>
        <div class="value">{top_team_name}<br><span class="hint">Total: {top_team_score:.2f}</span></div>
      </div>
      <div class="kpi">
        <div class="label">What to look for</div>
        <div class="value" style="font-size:14px;font-weight:700">
          Right side & up = <span style="color:#fbbf24">worst</span><br>
          High x = <span style="color:#34d399">most frequent</span><br>
          <span class="hint">Scatter bubbles scale by total choke</span>
        </div>
      </div>
    </div>

    <div class="card">{fig1.to_html(full_html=False, include_plotlyjs="cdn", config=config)}</div>

    <div class="grid2">
      <div class="card">{fig2.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
      <div class="card">{fig3.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
    </div>

    <div class="card">{fig4.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>

    <div class="grid2">
      <div class="card">{fig5.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
      <div class="card">{fig6.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
    </div>

    <div class="footer">
      Generated by <b>make_dashboard.py</b>. (If logos don’t show, your machine likely generated without internet access — charts still work.)
    </div>
  </div>
</body>
</html>
"""

    # Write the full dashboard to BOTH files so Pages loads it immediately at /
    with open(INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Dashboard written to:")
    print("   ", INDEX_HTML)
    print("   ", DASHBOARD_HTML)
    print("\nOpen locally:")
    print("   open docs/index.html")


if __name__ == "__main__":
    main()
