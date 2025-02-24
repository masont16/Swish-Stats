import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.endpoints import leaguegamelog as box
from nba_api.stats.endpoints import leaguedashteamstats as team

end = scoreboardv2.ScoreboardV2()
matches = end.get_dict()['resultSets'][0]['rowSet']
logs = []
def abb(ab):
    team = teams.find_team_by_abbreviation(ab)
    if team:
        return team['id']
    return None
for season in [2022, 2023, 2024]:
    logs.append(box.LeagueGameLog(season=season).get_data_frames()[0])
df = pd.concat(logs, ignore_index=True)
df['OPPONENT'] = df['MATCHUP'].str[-3:]
df['OPPONENT'] = df['OPPONENT'].apply(abb)
df['HOME'] = np.where(df['MATCHUP'].str[4] == '@', 1, 0)
df = df.drop(['TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'MIN', 'VIDEO_AVAILABLE'], axis=1)
df['WL'] = np.where(df['WL'] == 'W', 1, 0)
predictions = []
for match in matches:
    homeId = match[6]
    awayId = match[7]
    homeLogs = df[df['TEAM_ID'] == homeId]
    awayLogs = df[df['TEAM_ID'] == awayId]
    homeLogs['Last10'] = homeLogs['WL'].rolling(window=10, min_periods=1).mean()
    awayLogs['Last10'] = awayLogs['WL'].rolling(window=10, min_periods=1).mean()
    homeLogs['Last5'] = homeLogs['WL'].rolling(window=5, min_periods=1).mean()
    awayLogs['Last5'] = awayLogs['WL'].rolling(window=5, min_periods=1).mean()
    homeLogs['Last20'] = homeLogs['WL'].rolling(window=20, min_periods=1).mean()
    awayLogs['Last20'] = awayLogs['WL'].rolling(window=20, min_periods=1).mean()
    stats = team.LeagueDashTeamStats().get_data_frames()[0]
    home = stats[stats['TEAM_ID'] == homeId]
    away = stats[stats['TEAM_ID'] == awayId]
    homeToday = [home['FGM']/home['GP'], home['FGA']/home['GP'], home['FG_PCT']/home['GP'], home['FG3M']/home['GP'], home['FG3A']/home['GP'], home['FG3_PCT']/home['GP'], home['FTM']/home['GP'], home['FTA']/home['GP'], home['FT_PCT']/home['GP'], home['OREB']/home['GP'], home['DREB']/home['GP'], home['REB']/home['GP'], home['AST']/home['GP'], home['STL']/home['GP'], home['BLK']/home['GP'], home['TOV']/home['GP'], home['PF']/home['GP'],home['PLUS_MINUS']/home['GP'], awayId, 1, homeLogs['Last10'].iloc[-1], homeLogs['Last5'].iloc[-1], homeLogs['Last20'].iloc[-1]]
    awayToday = [away['FGM']/away['GP'], away['FGA']/away['GP'], away['FG_PCT']/away['GP'], away['FG3M']/away['GP'], away['FG3A']/away['GP'], away['FG3_PCT']/away['GP'], away['FTM']/away['GP'], away['FTA']/away['GP'], away['FT_PCT']/away['GP'], away['OREB']/away['GP'], away['DREB']/away['GP'], away['REB']/away['GP'], away['AST']/away['GP'], away['STL']/away['GP'], away['BLK']/away['GP'], away['TOV']/away['GP'], away['PF']/away['GP'],away['PLUS_MINUS']/away['GP'], homeId, 0, awayLogs['Last10'].iloc[-1], awayLogs['Last5'].iloc[-1], awayLogs['Last20'].iloc[-1]]
    features = homeLogs.drop(columns=['SEASON_ID', 'TEAM_ID', 'WL', 'PTS'])
    target = homeLogs['PTS']
    xTrain, xTest, yTrain, yTest = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(xTrain, yTrain)
    pred = pd.DataFrame([homeToday])
    clt = model.predict(pred)
    score = []
    score.append(teams.find_team_name_by_id(homeId)['full_name'])
    score.append(clt)
    features = awayLogs.drop(columns=['SEASON_ID', 'TEAM_ID', 'WL', 'PTS'])
    target = awayLogs['PTS']
    xTrain, xTest, yTrain, yTest = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(xTrain, yTrain)
    pred = pd.DataFrame([awayToday])
    clt = model.predict(pred)
    score.append(teams.find_team_name_by_id(awayId)['full_name'])
    score.append(clt)
    predictions.append(score)
df = pd.DataFrame.from_dict(predictions)
pd.set_option('display.max_columns', None)
print(df)