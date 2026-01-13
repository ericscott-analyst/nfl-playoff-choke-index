import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nfl_data_py as nfl

# ----------------------------
# Config
# ----------------------------
START_SEASON = 2000
END_SEASON = 2025
SEASONS = list(range(START_SEASON, END_SEASON + 1))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Game importance
IMPORTANCE_WEIGHTS = {
    "WC": 1.0,
    "DIV": 1.3,
    "CONF": 1.7,
    "SB": 2.2
}

# Late-game weighting (dominates the model)
LATE_WEIGHTS = {
    "NONE": 1.0,
    "Q4": 2.0,
    "LAST5": 3.5,
    "LAST2": 5.0,
    "OT_MULT": 1.25
}

# Special teams disaster weights
ST_WEIGHTS = {
    "MISSED_FG": 2.5,
    "MISSED_XP": 1.5,
    "ST_FUMBLE": 3.5,
    "ONSIDE_FAIL": 3.0
}


def num(x):
    return pd.to_numeric(x, errors="coerce")


# ----------------------------
# Playoff filter
# ----------------------------
def filter_playoffs(pbp):
    cols = set(pbp.columns)

    if "playoff" in cols:
        pbp = pbp[pbp["playoff"].fillna(0).astype(int) == 1].copy()
    elif "season_type" in cols:
        pbp = pbp[pbp["season_type"].astype(str).str.upper().isin(["POST", "PLAYOFF"])].copy()
    elif "game_type" in cols:
        pbp = pbp[pbp["game_type"].astype(str).str.upper().isin(["POST", "PLAYOFF"])].copy()
    else:
        raise ValueError("Cannot detect playoff games in this dataset.")

    print("Playoff play-by-play rows:", len(pbp))
    return pbp


# ----------------------------
# Infer playoff round
# ----------------------------
def infer_round(meta):
    if "week" not in meta.columns:
        meta["round"] = "UNK"
        meta["importance_weight"] = 1.0
        return meta

    meta["week"] = num(meta["week"])
    out = []
    rounds = ["WC", "DIV", "CONF", "SB"]

    for season, g in meta.groupby("season"):
        weeks = sorted(g["week"].dropna().unique())
        mapping = {}
        for i, w in enumerate(weeks[:4]):
            mapping[w] = rounds[i]

        gg = g.copy()
        gg["round"] = gg["week"].map(mapping).fillna("UNK")
        gg["importance_weight"] = gg["round"].map(IMPORTANCE_WEIGHTS).fillna(1.0)
        out.append(gg)

    return pd.concat(out, ignore_index=True)


# ----------------------------
# Build team-game table
# ----------------------------
def build_team_games(pbp):
    pbp = pbp.sort_values(["game_id", "play_id"]).copy()

    pbp["total_home_score"] = num(pbp["total_home_score"])
    pbp["total_away_score"] = num(pbp["total_away_score"])
    pbp["home_lead"] = pbp["total_home_score"] - pbp["total_away_score"]

    pbp["qtr"] = num(pbp.get("qtr", 0))
    pbp["game_seconds_remaining"] = num(pbp.get("game_seconds_remaining", np.nan))

    # Detect OT
    ot = pbp.groupby("game_id")["qtr"].max().reset_index(name="max_qtr")
    ot["went_ot"] = (ot["max_qtr"] >= 5).astype(int)

    # Base game stats
    games = pbp.groupby("game_id").agg(
        home_team=("home_team", "last"),
        away_team=("away_team", "last"),
        max_home_lead=("home_lead", "max"),
        max_away_lead=("home_lead", lambda s: (-s).max()),
        final_home=("total_home_score", "max"),
        final_away=("total_away_score", "max")
    ).reset_index()

    games["home_win"] = (games["final_home"] > games["final_away"]).astype(int)
    games["away_win"] = 1 - games["home_win"]

    games = games.merge(ot[["game_id", "went_ot"]], on="game_id", how="left")

    # Late game leads
    q4 = pbp[pbp["qtr"] == 4]
    last5 = q4[q4["game_seconds_remaining"] <= 300]
    last2 = q4[q4["game_seconds_remaining"] <= 120]

    q4_lead = q4.groupby("game_id")["home_lead"].max().reset_index(name="q4_lead")
    last5_lead = last5.groupby("game_id")["home_lead"].max().reset_index(name="last5_lead")
    last2_lead = last2.groupby("game_id")["home_lead"].max().reset_index(name="last2_lead")

    games = games.merge(q4_lead, on="game_id", how="left")
    games = games.merge(last5_lead, on="game_id", how="left")
    games = games.merge(last2_lead, on="game_id", how="left")

    games.fillna(0, inplace=True)

    # Team rows
    home = games[[
        "game_id", "home_team", "away_team",
        "max_home_lead", "final_home", "final_away",
        "home_win", "went_ot", "q4_lead", "last5_lead", "last2_lead"
    ]].copy()

    home.columns = [
        "game_id", "team", "opp",
        "max_lead", "points_for", "points_against",
        "win", "went_ot", "q4_lead", "last5_lead", "last2_lead"
    ]

    away = games[[
        "game_id", "away_team", "home_team",
        "max_away_lead", "final_away", "final_home",
        "away_win", "went_ot", "q4_lead", "last5_lead", "last2_lead"
    ]].copy()

    away.columns = [
        "game_id", "team", "opp",
        "max_lead", "points_for", "points_against",
        "win", "went_ot", "q4_lead", "last5_lead", "last2_lead"
    ]

    team_games = pd.concat([home, away], ignore_index=True)

    team_games["loss"] = (team_games["win"] == 0).astype(int)
    team_games["max_lead"] = team_games["max_lead"].clip(lower=0)
    team_games["q4_lead"] = team_games["q4_lead"].clip(lower=0)
    team_games["last5_lead"] = team_games["last5_lead"].clip(lower=0)
    team_games["last2_lead"] = team_games["last2_lead"].clip(lower=0)

    return team_games


# ----------------------------
# Late weight
# ----------------------------
def late_weight(row):
    if row["last2_lead"] > 0:
        w = LATE_WEIGHTS["LAST2"]
    elif row["last5_lead"] > 0:
        w = LATE_WEIGHTS["LAST5"]
    elif row["q4_lead"] > 0:
        w = LATE_WEIGHTS["Q4"]
    else:
        w = LATE_WEIGHTS["NONE"]

    if row["went_ot"] == 1 and row["q4_lead"] > 0:
        w *= LATE_WEIGHTS["OT_MULT"]

    return w


# ----------------------------
# Special teams meltdowns
# ----------------------------
def special_teams_scores(pbp):
    df = pbp.copy()
    df["desc"] = df.get("desc", "").astype(str)
    df["play_type"] = df.get("play_type", "").astype(str)

    # Missed FG
    missed_fg = df["desc"].str.contains("field goal", case=False, na=False) & \
                df["desc"].str.contains("no good|missed|blocked", case=False, na=False)

    # Missed XP
    missed_xp = df["desc"].str.contains("extra point", case=False, na=False) & \
                df["desc"].str.contains("no good|missed|blocked|failed", case=False, na=False)

    # ST fumbles
    fumble_lost = num(df.get("fumble_lost", 0)).fillna(0)
    st_fumble = (df["play_type"].isin(["punt", "kickoff"])) & (fumble_lost == 1)

    # Onside
    onside_attempt = df["desc"].str.contains("onside", case=False, na=False)
    recovered_by = df["desc"].str.contains("recovered by", case=False, na=False)
    onside_fail = onside_attempt & recovered_by

    st = pd.DataFrame({
        "game_id": df["game_id"],
        "team": df.get("posteam"),
        "missed_fg": missed_fg.astype(int),
        "missed_xp": missed_xp.astype(int),
        "st_fumble": st_fumble.astype(int),
        "onside_fail": onside_fail.astype(int)
    })

    st = st.dropna(subset=["team"])

    st["st_score"] = (
        st["missed_fg"] * ST_WEIGHTS["MISSED_FG"] +
        st["missed_xp"] * ST_WEIGHTS["MISSED_XP"] +
        st["st_fumble"] * ST_WEIGHTS["ST_FUMBLE"] +
        st["onside_fail"] * ST_WEIGHTS["ONSIDE_FAIL"]
    )

    return st.groupby(["game_id", "team"]).sum().reset_index()


# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading data...")
    pbp = nfl.import_pbp_data(SEASONS, downcast=True, cache=False)

    pbp["season"] = num(pbp.get("season"))
    pbp["week"] = num(pbp.get("week"))

    pbp = filter_playoffs(pbp)

    team_games = build_team_games(pbp)

    meta = pbp.groupby("game_id").agg(
        season=("season", "last"),
        week=("week", "last")
    ).reset_index()

    meta = infer_round(meta)

    team_games = team_games.merge(meta[["game_id", "season", "week", "round", "importance_weight"]],
                                  on="game_id", how="left")

    team_games["late_weight"] = team_games.apply(late_weight, axis=1)

    st = special_teams_scores(pbp)
    team_games = team_games.merge(st, on=["game_id", "team"], how="left").fillna(0)

    team_games["true_choke_score"] = (
        team_games["max_lead"]
        * team_games["importance_weight"]
        * team_games["late_weight"]
        * team_games["loss"]
        + team_games["st_score"] * team_games["importance_weight"] * team_games["loss"]
    )

    team_games["true_choke_10"] = ((team_games["max_lead"] >= 10) & (team_games["loss"] == 1)).astype(int)

    print("Playoff team-games:", len(team_games))
    print("Total chokes:", (team_games["true_choke_score"] > 0).sum())

    # Outputs
    all_games = team_games.sort_values("true_choke_score", ascending=False)
    all_games.to_csv(os.path.join(OUTPUTS_DIR, "all_team_games_playoffs.csv"), index=False)
    all_games.head(50).to_csv(os.path.join(OUTPUTS_DIR, "top_50_choke_games.csv"), index=False)

    # Team rankings
    by_team = team_games.groupby("team").agg(
        playoff_games=("game_id", "count"),
        playoff_losses=("loss", "sum"),
        total_choke=("true_choke_score", "sum"),
        avg_choke_per_loss=("true_choke_score", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
        true_chokes_10=("true_choke_10", "sum")
    ).reset_index()

    by_team["choke_rate"] = by_team["true_chokes_10"] / by_team["playoff_losses"].replace(0, np.nan)

    by_team.to_csv(os.path.join(OUTPUTS_DIR, "team_choke_rankings.csv"), index=False)

    print("Done.")
    print("Outputs in:", OUTPUTS_DIR)


if __name__ == "__main__":
    main()

