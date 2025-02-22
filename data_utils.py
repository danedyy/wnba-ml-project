import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# Read the data from the csv files 



def load_data(): 
    csv_awardsPlayers = pd.read_csv('basketballPlayoffs/awards_players.csv')
    csv_coaches = pd.read_csv('basketballPlayoffs/coaches.csv')
    csv_playersTeams = pd.read_csv('basketballPlayoffs/players_teams.csv')
    csv_players = pd.read_csv('basketballPlayoffs/players.csv')
    csv_seriesPost = pd.read_csv('basketballPlayoffs/series_post.csv')
    csv_teamsPost = pd.read_csv('basketballPlayoffs/teams_post.csv')
    csv_teams = pd.read_csv('basketballPlayoffs/teams.csv')

    return csv_awardsPlayers, csv_coaches, csv_playersTeams, csv_players, csv_seriesPost, csv_teamsPost, csv_teams


def setup_seriesPost(csv): 
    # Remove lgIDWinner and lgIDLoser columns from the csv_seriesPost dataframe
    csv = csv.drop(columns=['lgIDWinner', 'lgIDLoser'])
    return csv

def setup_coaches(csv_coaches):
    #create an aux csv file
    csv_coaches_aux = csv_coaches.copy()

    # remove the coaches that have 0 wins and 0 losses  
    csv_coaches_aux = csv_coaches_aux.drop(csv_coaches_aux[(csv_coaches_aux['won'] == 0) & (csv_coaches_aux['lost'] == 0)].index)
    
    attributes_accumulative = ['won','lost', 'post_wins', 'post_losses']

    for attr in attributes_accumulative:
        csv_coaches_aux[attr] = csv_coaches_aux.groupby('coachID')[attr].transform(lambda group: group.shift(1).rolling(min_periods=2, window=2).sum().fillna(0))
    
    csv_coaches_aux['winRateConf'] = np.where((csv_coaches_aux['won'] + csv_coaches_aux['lost']) > 0,
                                        csv_coaches_aux['won'] / (csv_coaches_aux['won'] + csv_coaches_aux['lost']),
                                        0)

    csv_coaches_aux['winRatePost'] = np.where((csv_coaches_aux['post_wins'] + csv_coaches_aux['post_losses']) > 0,
                                        csv_coaches_aux['post_wins'] / (csv_coaches_aux['post_wins'] + csv_coaches_aux['post_losses']),
                                        0)    
    
    csv_coaches_aux.drop(columns=['won','lost','post_wins','post_losses'], inplace=True)

    return csv_coaches_aux

def getCoachRating(csv_coaches, csv_teams):
    # Step 1: Merge the coaches data with the teams data on tmID and year to get playoff information
    merged_data = csv_coaches.merge(csv_teams[['tmID', 'year', 'playoff']], on=['tmID', 'year'], how='left')

    csv_coaches.to_csv('new_coaches_aux.csv')

    # Step 2: Select relevant columns for modeling
    relevant_columns = ['winRateConf', 'winRatePost','playoff']  # Adjust to match relevant coach stats
    
    # Filter the DataFrame to keep only relevant columns
    filtered_data = merged_data[relevant_columns]

    # Step 3: Separate features (coach statistics) and target variable (playoff success)
    X = filtered_data.drop(columns=['playoff'])  # Features (all coach statistics)
    y = filtered_data['playoff']  # Target variable (team's playoff status)

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Step 5: List of models to evaluate
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    best_correlation = -1  # Initialize the best correlation value
    best_model = None  # Track the best model
    best_coach_ratings = None  # To store the ratings from the best model

    # Step 6: Train and evaluate multiple models
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Predict the coach ratings
        filtered_data['coach_rating'] = model.predict(X)

        # Step 7: Calculate the correlation between coach_rating and playoff column
        correlation = filtered_data['coach_rating'].corr(filtered_data['playoff'])
        print(f"{model_name} Correlation with Playoff: {correlation * 100:.2f}%")

        # Track the model with the best correlation
        if correlation > best_correlation:
            best_correlation = correlation
            best_model = model_name
            best_coach_ratings = filtered_data[['coach_rating']].copy()

    # Step 8: Normalize the best coach ratings between 0 and 99
    scaler = MinMaxScaler(feature_range=(0, 99))
    best_coach_ratings[['coach_rating']] = scaler.fit_transform(best_coach_ratings[['coach_rating']])

    # Step 9: Add the best coach ratings to the original merged DataFrame
    csv_coaches['coach_rating'] = best_coach_ratings['coach_rating']

    # Step 10: Return the DataFrame with the new coach rating column
    print(f"Best Model: {best_model} with Correlation: {best_correlation * 100:.2f}%")
    return csv_coaches

def setup_teams(csv_teams,csv_teamsPost): 
        # Remove lgID column from the csv_teams dataframe
        csv_teams = csv_teams.drop(columns=['lgID','divID','seeded','tmTRB','tmDRB','tmORB','opptmORB','opptmDRB','opptmTRB','homeW','homeL','awayW','awayL','confW','confL','attend','arena','name','franchID'])
        csv_teamsPost = csv_teamsPost.drop(columns=['lgID'])

        csv_merge = pd.merge(csv_teams,csv_teamsPost, on=['tmID','year'], how='left')

        csv_merge['W'] = csv_merge['W'].fillna(0)
        csv_merge['L'] = csv_merge['L'].fillna(0)

        attributes_mean = ["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","rank"]
        attributes_accumulative = ['W','L','won','lost']


        # Sort the dataframe by team and year for proper cumulative calculations
        csv_merge = csv_merge.sort_values(by=['tmID', 'year'])

        # Apply cumulative mean for the defined attributes
        for attr in attributes_mean:
            # Calculate cumulative mean using expanding to include all prior values
            csv_merge[attr] = csv_merge.groupby('tmID')[attr].transform(lambda group: group.shift(1).rolling(min_periods=2, window=2).mean().fillna(0))

        # Apply cumulative sum for the defined attributes
        for attr in attributes_accumulative:
            csv_merge[attr] = csv_merge.groupby('tmID')[attr].transform(lambda group: group.shift(1).rolling(min_periods=2, window=2).sum().fillna(0))

        csv_merge['winRateConf'] = np.where((csv_merge['won'] + csv_merge['lost']) > 0,
                                            csv_merge['won'] / (csv_merge['won'] + csv_merge['lost']),
                                            0)

        csv_merge['winRatePost'] = np.where((csv_merge['W'] + csv_merge['L']) > 0,
                                            csv_merge['W'] / (csv_merge['W'] + csv_merge['L']),
                                            0)

        csv_merge['playoff'] = csv_merge['playoff'].replace({'Y':1,'N':0})

        csv_merge.drop(columns=['W','L','won','lost'], inplace=True)


        #from the attributes array get all the columns that either end with m or a 
        attributes = [attr for attr in attributes_mean if attr.endswith('m') or attr.endswith('a')]

        # Step 1: Get all the columns that either end with 'm' (made) or 'a' (attempted)
        attributes_m = [attr for attr in attributes_mean if attr.endswith('m')]
        attributes_a = [attr for attr in attributes_mean if attr.endswith('a')]

        # Step 2: Create the ratio columns where the ratio is calculated as the element that ends with 'm' divided by the element that ends with 'a'
        for attr_m in attributes_m:
            # Find the corresponding 'a' column by replacing 'm' with 'a' in the column name
            attr_a = attr_m[:-1] + 'a'
            
            if attr_a in attributes_a:
                # Create the new ratio column name, for example 'o_fg_pct' for 'o_fgm' and 'o_fga'
                ratio_col = f'{attr_m[:-1]}_pct'
                
                # Calculate the ratio (made / attempted) and handle division by 0 safely
                csv_merge[ratio_col] = np.where(csv_merge[attr_a] > 0, csv_merge[attr_m] / csv_merge[attr_a], 0)
        
        csv_merge.drop(columns=attributes, inplace=True)


        csv_merge['o_oreb_pct'] = np.where(csv_merge['o_reb'] > 0, csv_merge['o_oreb'] / csv_merge['o_reb'], 0)
        csv_merge['o_dreb_pct'] = np.where(csv_merge['o_reb'] > 0, csv_merge['o_dreb'] / csv_merge['o_reb'], 0)

        csv_merge['d_oreb_pct'] = np.where(csv_merge['d_reb'] > 0, csv_merge['d_oreb'] / csv_merge['d_reb'], 0)
        csv_merge['d_dreb_pct'] = np.where(csv_merge['d_reb'] > 0, csv_merge['d_dreb'] / csv_merge['d_reb'], 0)

        csv_merge['asts_to_pct'] = np.where(csv_merge['o_to'] > 0, csv_merge['o_asts'] / csv_merge['o_to'], 0) 
        csv_merge['stl_to_pct'] = np.where(csv_merge['o_to'] > 0, csv_merge['o_stl'] / csv_merge['o_to'], 0)    

        csv_merge['d_asts_to_pct'] = np.where(csv_merge['d_to'] > 0, csv_merge['d_asts'] / csv_merge['d_to'], 0)
        csv_merge['d_stl_to_pct'] = np.where(csv_merge['d_to'] > 0, csv_merge['d_stl'] / csv_merge['d_to'], 0)

        csv_merge.drop(columns=['o_reb','o_dreb','o_oreb','d_oreb','d_reb','o_to','o_asts','o_stl','d_to','d_asts','d_to','d_stl'], inplace=True)

        csv_merge['post_rank'] = np.where(csv_merge['firstRound'] == 'L', 8,
                                        np.where((csv_merge['firstRound'] == 'W') & (csv_merge['semis'] == 'L'), 4,
                                                np.where((csv_merge['firstRound'] == 'W') & (csv_merge['semis'] == 'W') & (csv_merge['finals'] == 'L'), 2,
                                                        np.where((csv_merge['firstRound'] == 'W') & (csv_merge['semis'] == 'W') & (csv_merge['finals'] == 'W'), 1, 0))))



        csv_merge['post_rank'] = csv_merge['post_rank'].replace(0, 9)

        # Correctly calculate the cumulative mean for post_rank
        csv_merge['post_rank_cummean'] = csv_merge.groupby('tmID')['post_rank'].transform(
            lambda group: group.shift(1).rolling(window=2, min_periods=2).mean().fillna(0)
)

        csv_merge.drop(columns=['firstRound','semis','finals'], inplace=True)



        return csv_merge

def setup_players(csv_players, csv_playersTeams): 
    # Remove the first season and last season attributes 
    csv_players = csv_players.drop(columns=['firstseason', 'lastseason'])
    # Remove the players that are not assigned to any team 
    players = csv_players['bioID'].unique()
    playersTeams = csv_playersTeams['playerID'].unique()
    csv_players.drop(csv_players[~csv_players['bioID'].isin(playersTeams)].index, inplace=True)

    return csv_players

def setup_players_mean(csv_players): 

    g_players = csv_players[(csv_players['pos'] == 'G') & csv_players['weight'] != 0]['weight'].mean()
    c_players  = csv_players[(csv_players['pos'] == 'C') & csv_players['weight'] != 0]['weight'].mean()
    f_players = csv_players[(csv_players['pos'] == 'F') & csv_players['weight'] != 0]['weight'].mean()
    c_f_players = csv_players[(csv_players['pos'] == 'C-F') & csv_players['weight'] != 0]['weight'].mean()
    g_f_players = csv_players[(csv_players['pos'] == 'G-F') & csv_players['weight'] != 0]['weight'].mean()
    f_c_players = csv_players[(csv_players['pos'] == 'F-C') & csv_players['weight'] != 0]['weight'].mean()
    f_g_players = csv_players[(csv_players['pos'] == 'F-G') & csv_players['weight'] != 0]['weight'].mean()

    csv_players.loc[(csv_players['pos'] == 'G') & (csv_players['weight'] == 0), 'weight'] = int(g_players)
    csv_players.loc[(csv_players['pos'] == 'C') & (csv_players['weight'] == 0), 'weight'] = int(c_players)
    csv_players.loc[(csv_players['pos'] == 'F') & (csv_players['weight'] == 0), 'weight'] = int(f_players)
    csv_players.loc[(csv_players['pos'] == 'C-F') & (csv_players['weight'] == 0), 'weight'] = int(c_f_players)
    csv_players.loc[(csv_players['pos'] == 'G-F') & (csv_players['weight'] == 0), 'weight'] = int(g_f_players)
    csv_players.loc[(csv_players['pos'] == 'F-C') & (csv_players['weight'] == 0), 'weight'] = int(f_c_players)
    csv_players.loc[(csv_players['pos'] == 'F-G') & (csv_players['weight'] == 0), 'weight'] = int(f_g_players)

    g_height = csv_players[(csv_players['pos'] == 'G') & csv_players['height'] != 0]['height'].mean()
    c_height  = csv_players[(csv_players['pos'] == 'C') & csv_players['height'] != 0]['height'].mean()
    f_height = csv_players[(csv_players['pos'] == 'F') & csv_players['height'] != 0]['height'].mean()
    c_f_height = csv_players[(csv_players['pos'] == 'C-F') & csv_players['height'] != 0]['height'].mean()
    g_f_height = csv_players[(csv_players['pos'] == 'G-F') & csv_players['height'] != 0]['height'].mean()
    f_c_height = csv_players[(csv_players['pos'] == 'F-C') & csv_players['height'] != 0]['height'].mean()
    f_g_height = csv_players[(csv_players['pos'] == 'F-G') & csv_players['height'] != 0]['height'].mean()

    csv_players.loc[(csv_players['pos'] == 'G') & (csv_players['height'] == 0), 'height'] = int(g_height)
    csv_players.loc[(csv_players['pos'] == 'C') & (csv_players['height'] == 0), 'height'] = int(c_height)
    csv_players.loc[(csv_players['pos'] == 'F') & (csv_players['height'] == 0), 'height'] = int(f_height)
    csv_players.loc[(csv_players['pos'] == 'C-F') & (csv_players['height'] == 0), 'height'] = int(c_f_height)
    csv_players.loc[(csv_players['pos'] == 'G-F') & (csv_players['height'] == 0), 'height'] = int(g_f_height)
    csv_players.loc[(csv_players['pos'] == 'F-C') & (csv_players['height'] == 0), 'height'] = int(f_c_height)
    csv_players.loc[(csv_players['pos'] == 'F-G') & (csv_players['height'] == 0), 'height'] = int(f_g_height)
    return csv_players

def setup_players_algorithm(csv_players):
    # Count and display the number of rows with missing weights
    print(f"Number of missing weights: {(csv_players['weight'] == 0).sum()}")

    # Selecting relevant columns
    important_columns = ['pos', 'height', 'weight']
    players_data = csv_players[important_columns]

    # Encode the 'pos' column into dummy variables
    players_data = pd.get_dummies(players_data, columns=['pos'])

    # Split data into rows with known and missing weights
    known_weights = players_data[players_data['weight'] != 0]
    missing_weights = players_data[players_data['weight'] == 0]

    # Define training and testing sets
    X_train = known_weights.drop(columns=['weight'])
    y_train = known_weights['weight']
    X_test = missing_weights.drop(columns=['weight'])

    # Set up models with initial hyperparameters for each
    models = {
        'KNN': KNeighborsRegressor(),
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Evaluate models using cross-validation and select the best one
    best_model = None
    best_score = -np.inf
    for name, model in models.items():
        # Perform cross-validation for each model
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        avg_score = np.mean(scores)
        print(f"{name} model average MSE: {-avg_score}")
        
        # Select the model with the highest average cross-validation score
        if avg_score > best_score:
            best_score = avg_score
            best_model = model

    print(f"Best model selected: {best_model}")

    # Fine-tune the best model with grid search (if applicable)
    if isinstance(best_model, KNeighborsRegressor):
        # For KNN, optimize the n_neighbors parameter
        initial_n_neighbors = int(np.sqrt(len(X_train)))
        param_grid = {'n_neighbors': range(1, initial_n_neighbors + 1)}
        grid_search = GridSearchCV(best_model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best KNN n_neighbors: {grid_search.best_params_['n_neighbors']}")
    elif isinstance(best_model, DecisionTreeRegressor) or isinstance(best_model, RandomForestRegressor):
        # For tree-based models, optimize the max_depth parameter
        param_grid = {'max_depth': range(1, 21)}
        grid_search = GridSearchCV(best_model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best tree depth: {grid_search.best_params_['max_depth']}")

    # Fit the best model to known weights and predict missing weights
    best_model.fit(X_train, y_train)
    predicted_weights = best_model.predict(X_test)

    # Fill in the missing weights in the original dataframe
    csv_players.loc[csv_players['weight'] == 0, 'weight'] = predicted_weights

    # Output the number of filled missing weights
    print(f"Number of weights predicted: {len(predicted_weights)}")

    return csv_players


def setup_playersTeams(csv_playersTeams,csv_awardsPlayers):
    csv_playersTeams = csv_playersTeams.drop(columns=['lgID'])
    csv_playersTeams = csv_playersTeams.drop(csv_playersTeams[csv_playersTeams['GP'] == 0].index)
    csv_playersTeams = csv_playersTeams.drop(csv_playersTeams[csv_playersTeams['minutes'] == 0].index)

    attributes_mean = ["GP","GS","minutes","points","oRebounds","dRebounds","rebounds","assists","steals","blocks","turnovers","PF","fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted","threeMade","dq","PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds","PostdRebounds","PostRebounds","PostAssists","PostSteals","PostBlocks","PostTurnovers","PostPF","PostfgAttempted","PostfgMade","PostftAttempted","PostftMade","PostthreeAttempted","PostthreeMade","PostDQ"]

    # Apply the cummulative sum for the defined attributes 
    
    for attr in attributes_mean:

        csv_playersTeams[attr] = csv_playersTeams.groupby('playerID')[attr].transform(lambda group: group.shift(1).rolling(min_periods=2, window=2).mean().fillna(0))
    
    # Create a column named player_awards_count that counts the number of awards each player has won

    csv_playersTeams['player_award_count'] = 0

    # Initialize a dictionary to keep track of total awards won by each player
    player_award_total = {}

    for index, row in csv_awardsPlayers.iterrows():
        player_id = row['playerID']
        
        if player_id not in player_award_total:
            player_award_total[player_id] = 0
        
        player_award_total[player_id] += 1
        
        csv_playersTeams.loc[
            (csv_playersTeams['playerID'] == player_id) & 
            (csv_playersTeams['year'] == row['year'] + 1),
            'player_award_count'
        ] = player_award_total[player_id] 



    return csv_playersTeams

def getTeamRating(csv_teams):
    # Create a copy of the DataFrame
    csv_teams_copy = csv_teams.copy()  

    # Drop the columns that are not relevant for the target
    csv_teams_copy = csv_teams_copy.drop(columns=['year', 'confID', 'tmID'])

    # Step 3: Calculate correlation with the 'playoff' column
    correlation_with_playoff = csv_teams_copy.corr()['playoff'].drop('playoff')  # Drop self-correlation

    # Calculate the total correlation for normalization (using raw values)
    total_correlation = correlation_with_playoff.sum()  # Sum of raw correlations

    # Step 4: Calculate weights based on correlation
    weights = {feature: correlation / total_correlation for feature, correlation in correlation_with_playoff.items()}

    # Step 5: Calculate team ratings based on the weights
    csv_teams_copy['team_rating'] = sum(weights[feature] * csv_teams_copy[feature] for feature in weights)

    # Optionally, apply exponential smoothing to the ratings for a rolling effect
    csv_teams_copy['team_rating'] = csv_teams_copy['team_rating'].ewm(span=3, adjust=False).mean()  # Use Exponential Moving Average

    # Step 6: Normalize the team ratings between 0 and 99
    min_rating = csv_teams_copy['team_rating'].min()
    max_rating = csv_teams_copy['team_rating'].max()
    
    # Apply min-max normalization
    csv_teams_copy['team_rating'] = ((csv_teams_copy['team_rating'] - min_rating) / (max_rating - min_rating)) * 99

    # Return the DataFrame with the new team rating column
    return csv_teams_copy


def getTeamRating2(csv_teams):
    # Create a copy of the DataFrame
    csv_teams_copy = csv_teams.copy()  

    # Separate features and target variable
    X = csv_teams_copy.drop(columns=['playoff'])  # Features
    y = csv_teams_copy['playoff']  # Target variable

    # Step 1: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # List of algorithms to test
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    best_correlation = -1  # Initialize the best correlation
    best_model = None  # Initialize the best model
    best_team_ratings = None  # To store the best ratings

    for model_name, model in models.items():
        # Step 2: Train the model
        model.fit(X_train, y_train)

        # Step 3: Predict the playoff outcome on the original dataset
        csv_teams_copy['team_rating'] = model.predict(X)

        # Step 4: Calculate the correlation with the playoff column
        correlation = csv_teams_copy['team_rating'].corr(csv_teams_copy['playoff'])

        print(f"{model_name} Correlation with Playoff: {correlation * 100:.2f}%")

        # Step 5: Check if this is the best model so far
        if correlation > best_correlation:
            best_correlation = correlation
            best_model = model_name
            best_team_ratings = csv_teams_copy[['team_rating']].copy()

    # Normalize the best team ratings between 0 and 99
    scaler = MinMaxScaler(feature_range=(0, 99))
    best_team_ratings[['team_rating']] = scaler.fit_transform(best_team_ratings[['team_rating']])

    # Add the best ratings to the original DataFrame
    csv_teams_copy['team_rating'] = best_team_ratings['team_rating']

    print(f"Best Model: {best_model} with Correlation: {best_correlation * 100:.2f}%")

    # Return the DataFrame with the new team rating column
    return csv_teams_copy


# Function to normalize award names
def normalize_award_name(award_name):
    # List of standard award names
    standard_awards = [
        'all-star game most valuable player', 'coach of the year', 'defensive player of the year',
        'kim perrot sportsmanship award', 'most improved player', 'most valuable player',
        'rookie of the year', 'sixth woman of the year', 'wnba all decade team honorable mention',
        'wnba all-decade team', 'wnba finals most valuable player'
    ]
    award_name = award_name.lower().replace(' awards', ' award').strip()
    # Check in the standard awards list if there is any exact match 
    for standard_award in standard_awards:
        if award_name == standard_award:
            return award_name
    else:  
        # Search for a similar award name in the standard awards list
        for standard_award in standard_awards:
            if award_name in standard_award:
                return standard_award


def setup_awardsPlayers(csv_awardsPlayers): 
    # Drop the lgID columns from the csv_awardsPlayers dataframe
    csv_awardsPlayers = csv_awardsPlayers.drop(columns=['lgID'])
    # replace all the names for lower case 
    csv_awardsPlayers['award'] = csv_awardsPlayers['award'].str.lower()
    return csv_awardsPlayers

def setup_teamsPost(csv_teamsPost): 
    # remove the lgID column from the csv_teamsPost dataframe
    csv_teamsPost = csv_teamsPost.drop(columns=['lgID'])
    return csv_teamsPost

def get_rating_per_player(csv_playersTeams,csv_players): 

    result_rows = []
    # Define weights for different positions
    position_weights = {
        'C': {
            'points': 1.0, 'rebounds': 1.5, 'assists': 0.5, 'steals': 0.5, 
            'blocks': 1.5, 'turnovers': -0.8, 'PF': -0.6, 'fgMade': 0.7, 
            'ftMade': 0.6, 'threeMade': 0.3
        },
        'F': {
            'points': 1.0, 'rebounds': 0.8, 'assists': 0.7, 'steals': 1.0, 
            'blocks': 1.0, 'turnovers': -0.7, 'PF': -0.5, 'fgMade': 0.6, 
            'ftMade': 0.7, 'threeMade': 0.6
        },
        'G': {
            'points': 1.0, 'rebounds': 0.5, 'assists': 1.2, 'steals': 1.5, 
            'blocks': 0.5, 'turnovers': -0.7, 'PF': -0.5, 'fgMade': 0.8, 
            'ftMade': 0.7, 'threeMade': 1.0
        },
        # Hybrid positions (averaged weights from both positions they represent)
        'F-C': {
            'points': 1.0, 'rebounds': 1.2, 'assists': 0.6, 'steals': 0.75, 
            'blocks': 1.25, 'turnovers': -0.75, 'PF': -0.55, 'fgMade': 0.65, 
            'ftMade': 0.65, 'threeMade': 0.45
        },
        'G-F': {
            'points': 1.0, 'rebounds': 0.65, 'assists': 0.95, 'steals': 1.25, 
            'blocks': 0.75, 'turnovers': -0.7, 'PF': -0.5, 'fgMade': 0.7, 
            'ftMade': 0.7, 'threeMade': 0.8
        },
        'F-G': {
            'points': 1.0, 'rebounds': 0.65, 'assists': 0.95, 'steals': 1.25, 
            'blocks': 0.75, 'turnovers': -0.7, 'PF': -0.5, 'fgMade': 0.7, 
            'ftMade': 0.7, 'threeMade': 0.8
        },
        'C-F': {
            'points': 1.0, 'rebounds': 1.2, 'assists': 0.6, 'steals': 0.75, 
            'blocks': 1.25, 'turnovers': -0.75, 'PF': -0.55, 'fgMade': 0.65, 
            'ftMade': 0.65, 'threeMade': 0.45
        }
    }
    
    # Merge the players and playersTeams dataframes on playerID, from the players csv only merge the position column
    players_rating = pd.merge(csv_playersTeams, csv_players[['bioID','pos']], left_on=['playerID'],right_on=['bioID'], how='inner')

    players_rating.drop(columns=['bioID'], inplace=True)


    players_rating_end = pd.DataFrame(columns=['playerID', 'rating_player', 'year', 'tmID', 'position'])
    
    # Iterate over each row in csv_playersTeams
    for index, row in players_rating.iterrows():
        # Get the player's position
        position = row['pos']
        
        # Get the weights based on the player's position, defaulting to Guard (G) if position not recognized
        weights = position_weights.get(position, position_weights['G'])
        
        # Ensure that 'GP' (games played) is not zero to avoid division by zero
        if row['GP'] == 0:
            rating = 0
        else:
            # Calculate the rating using position-specific weights
            rating = (
                row['points'] * weights['points'] +
                row['rebounds'] * weights['rebounds'] +
                row['assists'] * weights['assists'] +
                row['steals'] * weights['steals'] +
                row['blocks'] * weights['blocks'] +
                row['turnovers'] * weights['turnovers'] +  # Turnovers penalized
                row['PF'] * weights['PF'] +               # Personal fouls penalized
                row['fgMade'] * weights['fgMade'] +
                row['ftMade'] * weights['ftMade'] +
                row['threeMade'] * weights['threeMade']
            ) / row['GP']
        
        result_rows.append({
            'playerID': row['playerID'],
            'rating_player': rating,
            'year': row['year'],
            'tmID': row['tmID'],
            'position': position
        })
    
    # Convert the result list to a dataframe
    players_rating_end = pd.DataFrame(result_rows)

    return players_rating_end


def getPlayerRating(csv_playersTeams, csv_players, csv_teams):
    # Merge the player statistics with the playoff information from teams
    merged_data = csv_playersTeams.merge(csv_teams[['tmID', 'year', 'playoff']], on=['tmID', 'year'], how='left')

    # Select relevant columns for modeling
    relevant_columns = ['GP', 'GS', 'minutes', 'points', 'oRebounds', 'dRebounds', 'rebounds', 
                        'assists', 'steals', 'blocks', 'turnovers', 'PF', 
                        'fgAttempted', 'fgMade', 'ftAttempted', 'ftMade', 
                        'threeAttempted', 'threeMade', 'player_award_count', 'playoff']
    
    # Filter the DataFrame to keep only relevant columns
    filtered_data = merged_data[relevant_columns]

    # Separate features and target variable
    X = filtered_data.drop(columns=['playoff'])  # Features
    y = filtered_data['playoff']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # List of regression models to evaluate
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    best_correlation = -1  # Initialize the best correlation
    best_model = None  # Initialize the best model
    best_player_ratings = None  # To store the best ratings

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Predict player ratings using the model
        filtered_data['player_rating'] = model.predict(X)

        # Assess the correlation with the playoff column
        correlation = filtered_data['player_rating'].corr(filtered_data['playoff'])
        print(f"{model_name} Correlation with Playoff: {correlation * 100:.2f}%")

        # Check if this is the best model so far
        if correlation > best_correlation:
            best_correlation = correlation
            best_model = model_name
            best_player_ratings = filtered_data['player_rating']

    print(f"Best Model: {best_model} with Correlation: {best_correlation * 100:.2f}%")

    # Add the best player ratings to the original merged data
    merged_data['player_rating'] = best_player_ratings

    # Save the updated DataFrame with player ratings to a new CSV file
    merged_data.to_csv('output_csv_path', index=False)

    # Return the updated DataFrame
    return merged_data

    



def get_rating_per_team(csv_playersRatings_sorted,csv_teams): 
    teams_rating = pd.DataFrame(columns=['tmID', 'rating_team', 'year'])
    grouped = csv_playersRatings_sorted.groupby(['tmID', 'year']).agg({'rating_player': 'mean'}).reset_index()
    grouped.columns = ['tmID', 'year', 'rating_team']
    grouped = pd.merge(grouped,csv_teams[['tmID','confID']], on='tmID', how='left')
    return grouped


def get_rating_per_coach(csv_coaches, csv_teams):
    # Merge the coaches and teams dataframes on teamID and year
    merged_df = pd.merge(csv_coaches, csv_teams, on=['tmID', 'year'], how='left')
    
    # Check the columns of the merged dataframe
    print("Columns in merged_df:", merged_df.columns)
    
    # Create a new dataframe
    coaches_rating = pd.DataFrame(columns=['coachID', 'rating_coach', 'year','tmID','playoff'])
    
    # Iterate over each row in the merged dataframe
    for index, row in merged_df.iterrows():
        # Calculate the rating
        rating = 0
        rating += csv_coaches.at[index,'won'] * 2  # Wins are highly valuable
        rating -= csv_coaches.at[index,'lost'] * 2  # Losts are highly penalized
        if row['playoff'] == 'Y':
            rating += 5  # Playoff appearance adds to the rating
        
        # Assign values to the new dataframe
        coaches_rating.at[index, 'coachID'] = row['coachID']
        coaches_rating.at[index, 'rating_coach'] = rating
        coaches_rating.at[index, 'year'] = row['year']
        coaches_rating.at[index,'tmID'] = row['tmID']
        coaches_rating.at[index,'playoff'] = row['playoff']
    
    return coaches_rating

def rank_features_per_position(csv_playerTeams,csv_players):
    positions = ['G', 'C', 'F', 'C-F', 'G-F', 'F-C', 'F-G']
    statistics_features = ["points","oRebounds","dRebounds","rebounds","assists","steals","blocks","turnovers","PF","fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted","threeMade","dq"]
    return 1 



