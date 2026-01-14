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
            f"Make sure your outputs/*.csv exist (run build_choke_index.py first)."
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

        # Prefer ESPN logo; if missing, fallback
        if "team_logo_espn" not in meta.columns and "team_logo_wikipedia" in meta.columns:
            meta["team_logo_espn"] = meta["team_logo_wikipedia"]

        keep = [c for c in ["team", "team_conf", "team_logo_espn"] if c in meta.columns]
        return meta[keep].copy()
    except Exception:
        return None


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

    # Top 50 for headline charts
    games_top50 = games.sort_values("true_choke_score", ascending=False).head(50).copy()

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
    # Styling
    # -----------------------
    template = "plotly_dark"
    base_layout = dict(
        template=template,
        font=dict(size=14),
        margin=dict(l=26, r=26, t=72, b=26),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
    )
    config = {"responsive": True, "displaylogo": False}

    ROUND_COLORS = {
        "Wild Card": "#60a5fa",
        "Divisional": "#34d399",
        "Conference": "#fbbf24",
        "Super Bowl": "#f87171",
        "Unknown": "#9ca3af",
    }
    CONF_COLORS = {"AFC": "#60a5fa", "NFC": "#34d399", "Unknown": "#9ca3af"}

    # -----------------------
    # CHART 1: Top 25 chokes
    # -----------------------
    top25 = games_top50.head(25).iloc[::-1].copy()
    fig1 = px.bar(
        top25,
        x="true_choke_score",
        y="label",
        orientation="h",
        color="round_bucket",
        color_discrete_map=ROUND_COLORS,
        hover_data={
            "team": True,
            "opp": True,
            "round_bucket": True,
            "true_choke_score": ":.2f",
            "max_lead": True,
            "importance_weight": ":.2f",
            "late_weight": ":.2f",
        },
        template=template,
    )
    fig1.update_layout(
        **base_layout,
        height=920,
        legend_title_text="Round",
        title=dict(text="Top 25 Biggest Playoff Chokes (True Choke Score)", x=0.02, xanchor="left"),
    )
    fig1.update_yaxes(title="", automargin=True)
    fig1.update_xaxes(title="True Choke Score")

    # -----------------------
    # CHART 2: Choke MOST
    # -----------------------
    most_df = teams.sort_values("choke_rate", ascending=False).head(16).copy()
    fig2 = px.bar(
        most_df.iloc[::-1],
        x="choke_rate",
        y="team",
        orientation="h",
        color="team_conf",
        color_discrete_map=CONF_COLORS,
        hover_data={
            "choke_rate": ":.2f",
            "playoff_losses": True,
            "playoff_games": True,
            "total_choke": ":.2f",
            "avg_choke_per_loss": ":.2f",
        },
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

    # -----------------------
    # CHART 3: Choke WORST
    # -----------------------
    worst_df = teams.sort_values("avg_choke_per_loss", ascending=False).head(16).copy()
    fig3 = px.bar(
        worst_df.iloc[::-1],
        x="avg_choke_per_loss",
        y="team",
        orientation="h",
        color="team_conf",
        color_discrete_map=CONF_COLORS,
        hover_data={
            "avg_choke_per_loss": ":.2f",
            "playoff_losses": True,
            "total_choke": ":.2f",
            "choke_rate": ":.2f",
        },
        template=template,
    )
    fig3.update_layout(
        **base_layout,
        height=520,
        legend_title_text="Conf",
        title=dict(text="Teams That Choke WORST (Avg Severity per Loss)", x=0.02, xanchor="left"),
    )
    fig3.update_yaxes(title="", automargin=True)
    fig3.update_xaxes(title="Avg True Choke Score per Loss")

    # -----------------------
    # CHART 4: Scatter (logos in hover)
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
        height=700,
        showlegend=False,
        title=dict(text="Choke Most vs Choke Worst (Top 24 by Total Choke) — Logos in Hover", x=0.02, xanchor="left"),
    )
    fig4.update_xaxes(title="Choke Rate (Most)")
    fig4.update_yaxes(title="Avg Choke Severity per Loss (Worst)")

    # -----------------------
    # TEAM DEEP DIVE table data (ALL rows in your available games CSV)
    # -----------------------
    deep_df = games.copy()

    for col in ["season", "week"]:
        if col not in deep_df.columns:
            deep_df[col] = np.nan

    # pick columns that are likely present
    preferred_cols = ["season", "week", "round", "opp", "points_for", "points_against", "max_lead", "true_choke_score"]
    cols = [c for c in preferred_cols if c in deep_df.columns]

    deep_df = deep_df[["team"] + cols].copy()
    deep_df["team"] = deep_df["team"].astype(str).str.upper()
    deep_df["season"] = pd.to_numeric(deep_df["season"], errors="coerce")
    deep_df["week"] = pd.to_numeric(deep_df["week"], errors="coerce")

    deep_df["true_choke_score"] = pd.to_numeric(deep_df.get("true_choke_score", 0.0), errors="coerce").fillna(0.0)
    deep_df["max_lead"] = pd.to_numeric(deep_df.get("max_lead", 0.0), errors="coerce").fillna(0.0)

    # Sort default: most severe first per team
    deep_df = deep_df.sort_values(["team", "true_choke_score"], ascending=[True, False]).copy()

    team_list = sorted([t for t in deep_df["team"].dropna().unique().tolist() if str(t).strip() != ""])
    payload = {t: deep_df[deep_df["team"] == t].to_dict(orient="records") for t in team_list}
    payload_json = json.dumps(payload)

    # -----------------------
    # KPIs
    # -----------------------
    top_team_total = teams.sort_values("total_choke", ascending=False).head(1)
    top_team_name = top_team_total["team"].iloc[0] if len(top_team_total) else "N/A"
    top_team_score = float(top_team_total["total_choke"].iloc[0]) if len(top_team_total) else 0.0

    biggest_game = games_top50.iloc[0]["label"] if len(games_top50) else "N/A"
    biggest_score = float(games_top50.iloc[0]["true_choke_score"]) if len(games_top50) else 0.0

    build_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -----------------------
    # HTML (mobile-friendly)
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
      padding: 22px 16px 56px 16px;
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
      margin: 10px 0 0 0;
    }}
    .hint {{
      color: var(--muted);
      font-size: 12px;
      margin: 8px 0 0 0;
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
    .section-title {{
      font-weight: 800;
      font-size: 16px;
      margin: 0 0 8px 0;
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
      width: min(360px, 100%);
    }}
    .pill {{
      padding: 8px 10px;
      background: var(--chip);
      border: 1px solid var(--border);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
    }}
    .table-wrap {{
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: auto;
      max-height: 520px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px; /* horizontal scroll on mobile */
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
    .muted {{
      color: var(--muted);
    }}
    .footer {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 950px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 26px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>NFL Playoff True Choke Index (2000–2025)</h1>
    <p class="sub">
      Mobile-friendly sports analytics report. Team Deep Dive shows every available choke game and its score.
    </p>
    <p class="hint">Tip: Tap/click column headers to sort • Scroll sideways on mobile for all columns</p>

    <div class="kpis">
      <div class="kpi">
        <div class="label">#1 Choke Game</div>
        <div class="value" style="font-size:14px;font-weight:700">{biggest_game}<br><span class="muted">Score: {biggest_score:.2f}</span></div>
      </div>
      <div class="kpi">
        <div class="label">Top Franchise (Total Choke)</div>
        <div class="value">{top_team_name}<br><span class="muted">Total: {top_team_score:.2f}</span></div>
      </div>
      <div class="kpi">
        <div class="label">Build</div>
        <div class="value" style="font-size:14px;font-weight:700">{build_stamp}<br><span class="muted">If this timestamp changed, your site updated.</span></div>
      </div>
    </div>

    <div class="card">{fig1.to_html(full_html=False, include_plotlyjs="cdn", config=config)}</div>

    <div class="grid2">
      <div class="card">{fig2.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
      <div class="card">{fig3.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>
    </div>

    <div class="card">{fig4.to_html(full_html=False, include_plotlyjs=False, config=config)}</div>

    <div class="card">
      <div class="section-title">Team Deep Dive — All Chokes + Scores</div>
      <div class="controls">
        <select id="teamSelect"></select>
        <div class="pill" id="rowCount">0 games</div>
        <div class="pill">Click headers to sort</div>
      </div>

      <div class="table-wrap">
        <table id="teamTable">
          <thead><tr id="theadRow"></tr></thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>

      <div class="footer">
        Note: This table uses the data available in outputs/*.csv.
      </div>
    </div>
  </div>

<script>
  const DATA = {payload_json};
  const teams = Object.keys(DATA).sort();
  const defaultTeam = teams.includes("GB") ? "GB" : (teams[0] || "");

  const select = document.getElementById("teamSelect");
  const theadRow = document.getElementById("theadRow");
  const tbody = document.getElementById("tbody");
  const rowCount = document.getElementById("rowCount");

  const preferredCols = ["season","week","round","opp","points_for","points_against","max_lead","true_choke_score"];

  let currentTeam = defaultTeam;
  let sortKey = "true_choke_score";
  let sortDir = "desc";

  function fmt(val, key) {{
    if (val === null || val === undefined) return "";
    if (key === "true_choke_score") {{
      const n = Number(val);
      return isNaN(n) ? val : n.toFixed(2);
    }}
    if (key === "max_lead") {{
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

  function renderTable(team) {{
    currentTeam = team;
    const rows = DATA[team] || [];
    const cols = getCols(rows);

    // header
    theadRow.innerHTML = "";
    cols.forEach(col => {{
      const th = document.createElement("th");
      th.textContent = col.replaceAll("_"," ").toUpperCase();
      th.dataset.col = col;
      th.addEventListener("click", () => {{
        if (sortKey === col) {{
          sortDir = (sortDir === "asc") ? "desc" : "asc";
        }} else {{
          sortKey = col;
          sortDir = (col === "true_choke_score") ? "desc" : "asc";
        }}
        renderTable(currentTeam);
      }});
      theadRow.appendChild(th);
    }});

    // body
    const sorted = sortRows(rows);
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

    rowCount.textContent = `${{sorted.length}} games`;
  }}

  // populate select
  teams.forEach(t => {{
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    select.appendChild(opt);
  }});
  select.value = defaultTeam;

  select.addEventListener("change", (e) => {{
    sortKey = "true_choke_score";
    sortDir = "desc";
    renderTable(e.target.value);
  }});

  renderTable(defaultTeam);
</script>

</body>
</html>
"""

    # Write dashboard directly to index.html so your Pages URL opens the dashboard immediately
    with open(INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Dashboard written to:")
    print("   ", INDEX_HTML)
    print("   ", DASHBOARD_HTML)
    print("Build stamp:", build_stamp)
    print("\nOpen locally:")
    print("   open docs/index.html")


if __name__ == "__main__":
    main()
