import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)



# 
#                 scraping from ProFootballReference
# (pretty much untouched, this part normally works fine)
# 

def get_schedule(year=2024):
    url = f"https://www.pro-football-reference.com/years/{year}/games.htm"
    df_list = pd.read_html(url)
    games = df_list[0]

    #Remove duplicate header rows inserted into the body of the table
    games = games[games['Date'] != 'Date']

    #Rename columns
    games = games.rename(columns={
        'Winner/tie': 'Winner',
        'Loser/tie': 'Loser'
    })

    # Convert score columns to numeric safely
    for col in ['PtsW', 'PtsL']:
        games[col] = pd.to_numeric(games[col], errors='coerce')

    return games




# 
#   multi-year loading (for more data)
# 

def load_multiple_years(start=2018, end=2024):
    all_games = []
    all_teams = []

    for yr in range(start, end + 1):
        g = get_schedule(yr)
        t = get_team_stats(yr)

        g["Season"] = yr   # for rolling form later
        all_games.append(g)
        all_teams.append(t)

    return pd.concat(all_games, ignore_index=True), pd.concat(all_teams, ignore_index=True)



# 
#        rolling "last 3 games" team form calculation
# 

def add_last3_form(games):
    #ok this function was blowing up because PFR stores points as strings sometimes.
    #So let’s build the intermediate rows manually, THEN cast them numeric.

    form_rows = []

    for _, row in games.iterrows():
        teamW = row["Winner"]
        teamL = row["Loser"]
        ptsW  = row["PtsW"]
        ptsL  = row["PtsL"]
        season = row.get("Season", np.nan)

        # winner entry
        form_rows.append([teamW, season, ptsW, ptsL])
        # loser entry
        form_rows.append([teamL, season, ptsL, ptsW])

    form = pd.DataFrame(form_rows, columns=["Tm","Season","PtsFor","PtsAgainst"])

    #convert to numeric (the fix)
    form["PtsFor"] = pd.to_numeric(form["PtsFor"], errors="coerce")
    form["PtsAgainst"] = pd.to_numeric(form["PtsAgainst"], errors="coerce")
    

    #yeah sorting is needed for rolling
    form = form.sort_values(by=["Tm","Season"])

    #rolling mean of last 3 games
    form["last3_for"] = (
        form.groupby("Tm")["PtsFor"]
            .rolling(3, min_periods=1).mean()
            .reset_index(0, drop=True)
    )

    form["last3_against"] = (
        form.groupby("Tm")["PtsAgainst"]
            .rolling(3, min_periods=1).mean()
            .reset_index(0, drop=True)
    )

    return form



# 
#                   basic custom ELO
# 

def compute_elo(games, base_elo=1500, k=20):
    elos = {}

    for _, g in games.iterrows():
        home = g["Winner"] if g["PtsW"] > g["PtsL"] else g["Loser"]
        
        away = g["Loser"] if home == g["Winner"] else g["Winner"]

        eh = elos.get(home, base_elo)
        ea = elos.get(away, base_elo)

        exp_home = 1 / (1 + 10 ** ((ea - eh) / 400))
        actual = 1


        
        margin = abs(g["PtsW"] - g["PtsL"])
        change = k * (actual - exp_home) * (1 + margin / 14)

        elos[home] = eh + change
        elos[away] = ea - change

    #Convert to DataFrame: one row per team
    df = pd.DataFrame({
        "Team": list(elos.keys()),
        "ELO": list(elos.values())
    })

    return df


def expand_games(games):
    """
    Converts PFR's Winner/Loser format into the Tm/Opp format
    that get_team_stats() expects.
    """

    rows = []

    for _, g in games.iterrows():
        w = g["Winner"]
        l = g["Loser"]
        pw = g["PtsW"]
        pl = g["PtsL"]

        # Winner row
        rows.append({
            "Tm": w,
            "Opp": l,
            "PtsTm": pw,
            "PtsOpp": pl,
            "Season": g.get("Season", np.nan)
        })

        #Loser row
        rows.append({
            "Tm":l,
            "Opp": w,
            "PtsTm": pl,
            "PtsOpp": pw,
            
            "Season": g.get("Season", np.nan)
        })

    return pd.DataFrame(rows)


def get_team_stats(df):
    #Safely extract teams without forcing dtype conversion or full-frame copies
    tm_raw = df.get("Tm", [])
    opp_raw = df.get("Opp", [])

    # Filter out anything that isn’t a string (NaN, floats, None)
    teams_tm = [x for x in tm_raw if isinstance(x, str) and x.strip() != ""]
    teams_opp = [x for x in opp_raw if isinstance(x, str) and x.strip() != ""]

    # Merge and sort
    teams = sorted(set(teams_tm + teams_opp))

    rows = []
    for team in teams:
        # Filter rows ONLY where Tm matches, skipping NaN rows automatically

        
        team_games = df[df["Tm"] == team]
        if len(team_games) == 0:
            continue

        row = {
            "Team": team,
            "games": len(team_games),
            "points_for": team_games["PtsTm"].sum(),
            "points_against": team_games["PtsOpp"].sum(),
            "avg_for": team_games["PtsTm"].mean(),
            
            "avg_against": team_games["PtsOpp"].mean(),
        }
        rows.append(row)

    # RETURN AS DATAFRAME
    return pd.DataFrame(rows)


# 
# FIXED build_features
# 
def build_features(games, teams, year=2024):
    print("\n=== ENTER build_features ===")

    games = games.copy()
    games.columns = [str(c).strip() for c in games.columns]

    rename_map = {"Winner/tie": "Winner","Loser/tie": "Loser","PtsW":"PtsW","PtsL":"PtsL","Unnamed: 5":"HA"}
    games = games.rename(columns=rename_map)
    if "Date.1" in games.columns:
        games = games.drop(columns=["Date.1"])

    print("Columns after normalization:", games.columns.tolist())

    for col in ["Winner","Loser","PtsW","PtsL","HA"]:
        if col not in games.columns:
            print("Missing:", col)
            return None

    games["PtsW"] = pd.to_numeric(games["PtsW"], errors="coerce")
    games["PtsL"] = pd.to_numeric(games["PtsL"], errors="coerce")
    games = games.dropna(subset=["PtsW","PtsL"]).copy()
    games["Season"] = year

    games["home_team"] = np.where(games["HA"]=="@", games["Loser"], games["Winner"])
    games["away_team"] = np.where(games["HA"]=="@", games["Winner"], games["Loser"])
    games["home_win"]  = np.where(games["HA"]=="@", 0, 1)

    #rolling form
    try:
        form = add_last3_form(games)
    except Exception as e:
        print("\nERROR in add_last3_form:")
        print(e, "\n")
        return None

    print("Form preview:\n", form.head())

    #ELO
    try:
        elo_df = compute_elo(games)
    except Exception as e:
        print("\nERROR in compute_elo:")
        print(e, "\n")
        return None

    print("ELO preview:\n", elo_df.head())

    # erge rolling form
    games = games.merge(form[["Tm","last3_for","last3_against"]], left_on="home_team", right_on="Tm", how="left")
    games = games.rename(columns={"last3_for":"home_last3_for","last3_against":"home_last3_against"}).drop(columns=["Tm"])
    games = games.merge(form[["Tm","last3_for","last3_against"]], left_on="away_team", right_on="Tm", how="left")
    games = games.rename(columns={"last3_for":"away_last3_for","last3_against":"away_last3_against"}).drop(columns=["Tm"])

    #merge ELO
    games = games.merge(elo_df.rename(columns={"Team":"Winner","ELO":"WinnerELO"}), on="Winner", how="left")
    games = games.merge(elo_df.rename(columns={"Team":"Loser","ELO":"LoserELO"}), on="Loser", how="left")
    games["home_elo"] = np.where(games["home_team"]==games["Winner"], games["WinnerELO"], games["LoserELO"])

    
    games["away_elo"] = np.where(games["away_team"]==games["Winner"], games["WinnerELO"], games["LoserELO"])

    # merge team stats (teams is now a DataFrame)
    home_stats = teams.add_prefix("home_")
    away_stats = teams.add_prefix("away_")

    merged_rows = []
    for _, g in games.iterrows():
        h = home_stats[home_stats["home_Team"] == g["home_team"]]
        a = away_stats[away_stats["away_Team"] == g["away_team"]]

        if len(h)==1 and len(a)==1:
            row = pd.concat([
                g[[
                    "home_team","away_team","home_win",
                    "home_last3_for","home_last3_against",
                    "away_last3_for","away_last3_against",
                    "home_elo","away_elo"
                ]].to_frame().T.reset_index(drop=True),
                h.reset_index(drop=True),
                a.reset_index(drop=True)
            ], axis=1)
            merged_rows.append(row)

    df = pd.concat(merged_rows, ignore_index=True)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    
    df["form_pts_diff"] = df["home_last3_for"] - df["away_last3_for"]
    df["form_allowed_diff"] = df["home_last3_against"] - df["away_last3_against"]

    print("=== EXIT build_features ===")
    return df




# 
#                          TRAIN MODELS
# 

def train_models(df):

    df = df.copy()

    # convert all object columns to numeric when possible
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.replace("%","", regex=False)
            try:
                
                df[c] = pd.to_numeric(df[c])
            except:
                pass

    df = df.dropna(subset=["home_win"])

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric].drop(columns=["home_win"])
    y = df["home_win"]

    X = X.fillna(0)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    #logistic regression
    log = LogisticRegression(max_iter=1000)
    log.fit(Xtr_s, ytr)
    log_acc = accuracy_score(yte, log.predict(Xte_s))

    #random forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(Xtr, ytr)
    rf_acc = accuracy_score(yte, rf.predict(Xte))

    print(f"\nLogistic Accuracy: {log_acc:.3f}")
    print(f"RandomForest Accuracy: {rf_acc:.3f}\n")

    # Feature importance ----------------------------------------------------
    print("Top RF Features (help home team):")
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    print(imp.sort_values(ascending=False).head(10), "\n")

    print("Worst RF Features:")
    print(imp.sort_values(ascending=True).head(10), "\n")

    # logistic coefficients
    coef = pd.Series(log.coef_[0], index=X.columns)
    print("Top Logistic Magnitude:")
    print(coef.abs().sort_values(ascending=False).head(10), "\n")

    return log, rf, scaler


def predict_score(home_team, away_team, teams_df, elo_df):
    #Defaults
    default_stats = {"games":3,"points_for":63,"points_against":63,"avg_for":21,"avg_against":21}
    default_elo = 1500

    #Pull stats or defaults
    home_stats = teams_df[teams_df["Team"]==home_team].iloc[0] if home_team in teams_df["Team"].values else pd.Series(default_stats)
    away_stats = teams_df[teams_df["Team"]==away_team].iloc[0] if away_team in teams_df["Team"].values else pd.Series(default_stats)

      # ELO
    home_elo = float(elo_df[elo_df["Team"]==home_team]["ELO"].iloc[0]) if home_team in elo_df["Team"].values else default_elo
    away_elo = float(elo_df[elo_df["Team"]==away_team]["ELO"].iloc[0]) if away_team in elo_df["Team"].values else default_elo

    # Base points using averages
    home_base = home_stats["avg_for"]
    away_base = away_stats["avg_for"]

    # Adjust by ELO difference (simple linear scale)
    elo_diff = home_elo - away_elo
    home_adj = home_base + (elo_diff / 100)  # scaling factor
    away_adj = away_base - (elo_diff / 100)

    

    # Adjust by form differences
    form_diff = home_stats["avg_for"] - away_stats["avg_for"]
    home_score = home_adj + form_diff * 0.5
    away_score = away_adj - form_diff * 0.5

    return round(home_score), round(away_score)


# 
#                   PREDICT GAME FUNCTION
# 

def predict_game(home_team, away_team, teams_df, elo_df, log_model, rf_model, scaler, df_train_features):
    #Strip input
    home_team = home_team.strip()
    away_team = away_team.strip()

    #Default stats if missing (rpobably is)
    default_stats = {"games":3,"points_for":63,"points_against":63,"avg_for":21,"avg_against":21}
    
    default_elo = 1500
    default_form = 21

    # Pull team stats or default
    if home_team in teams_df['Team'].values:
        home_stats = teams_df[teams_df["Team"] == home_team].iloc[0]
    else:
        home_stats = pd.Series(default_stats)
    if away_team in teams_df['Team'].values:
        away_stats = teams_df[teams_df["Team"] == away_team].iloc[0]
    else:
        away_stats = pd.Series(default_stats)

    # ELO
    home_elo_row = elo_df[elo_df["Team"] == home_team]
    away_elo_row = elo_df[elo_df["Team"] == away_team]
    home_elo = float(home_elo_row["ELO"].iloc[0]) if not home_elo_row.empty else default_elo
    away_elo = float(away_elo_row["ELO"].iloc[0]) if not away_elo_row.empty else default_elo

    #Build full feature dict
    row = {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_games": home_stats["games"],
        "away_games": away_stats["games"],
        "home_points_for": home_stats["points_for"],
        "home_points_against": home_stats["points_against"],

        #ahhh
        "home_avg_for": home_stats["avg_for"],
        "home_avg_against": home_stats["avg_against"],
        "away_points_for": away_stats["points_for"],
        "away_points_against": away_stats["points_against"],
        "away_avg_for": away_stats["avg_for"],
        "away_avg_against": away_stats["avg_against"],
        "home_last3_for": default_form,
        "home_last3_against": default_form,
        "away_last3_for": default_form,
        "away_last3_against": default_form,
        "form_pts_diff": default_form,
        "form_allowed_diff": default_form
    }

    #Make DataFrame with the **same columns as training**
    X = pd.DataFrame([row])
    # Reorder columns to match training DataFrame
    X = X[[col for col in df_train_features if col in X.columns]]


    #Scale
    X_s = scaler.transform(X)

    #Predict
    log_pred = log_model.predict_proba(X_s)[0][1]
    rf_pred = rf_model.predict(X_s)[0]

    print(f"\nPredicted probability home team wins (Logistic Regression): {log_pred:.3f}")
    print(f"Predicted outcome (Random Forest): {'Away Wins' if rf_pred==1 else 'Home wins'}\n")
    home_pts, away_pts = predict_score(home_team, away_team, teams_df, elo_df)
    print(f"Predicted score: {home_team} {home_pts} - {away_team} {away_pts}")


# 
#                           MAIN BLOCK
# 


if __name__ == "__main__":
    year = 2024 #change for the year you want
    print(f"Loading NFL data for {year}...\n")

    # Load games and stats
    games = get_schedule(year)
    expanded = expand_games(games)
    teams = get_team_stats(expanded)

    # Build features and train
    df = build_features(games, teams, year)
    if df is None:
        print("Feature construction failed.")
        raise SystemExit

    log_model, rf_model, scaler = train_models(df)\
    
    df_train_features = df.drop(columns=["home_win"]).columns.tolist()

    #Show available teams
    print("\nAvailable teams:")
    for t in sorted(teams['Team'].tolist()):
        print("-", t)

    #Input teams
    home_team = input("\nEnter home team: ")
    away_team = input("Enter away team: ")

    predict_game(home_team, away_team, teams, compute_elo(games), log_model, rf_model, scaler, df_train_features)



  
#PLOTS
elo_df = compute_elo(games)
plt.figure(figsize=(12,6))
sns.barplot(x="Team", y="ELO", data=elo_df.sort_values("ELO", ascending=False))
plt.xticks(rotation=45)
plt.title("Current ELO Ratings by Team")
plt.ylabel("ELO")
plt.show()

teams_stats = teams.copy()
plt.figure(figsize=(12,6))
sns.scatterplot(x="avg_for", y="avg_against", data=teams_stats)
for i, row in teams_stats.iterrows():
    plt.text(row["avg_for"], row["avg_against"], row["Team"], fontsize=8)
plt.xlabel("Average Points For")
plt.ylabel("Average Points Against")
plt.title("Team Offensive vs Defensive Performance for 2024")
plt.show()


