import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import heapq # import heapq module

model = joblib.load('logreg_model_for_stanley_cup.pkl')

directory = 'C:\\Users\\Jeremy\\Dev\\Python\\logreg_model_for_stanley_cup\\nhl_data'

total_predictions = 0
total_correct_predictions = 0

def process_file(file):
    # This function reads and processes an Excel file and returns the real and predicted winning teams
    file_path = os.path.join(directory, file)
    new_statistics = pd.read_excel(file_path)
        
    columns_to_fill = ['Playoffs2R', 'Playoffs3R', 'Playoffs4R']

    new_statistics[columns_to_fill] = new_statistics[columns_to_fill].fillna(0)

    new_statistics_numeric = new_statistics.drop(['StanleyCup', 'Season', 'Team'], axis=1)
        
    scaler = StandardScaler()
    new_statistics_scaled = scaler.fit_transform(new_statistics_numeric)
        
    probabilities = model.predict_proba(new_statistics_scaled)
        
    teams = new_statistics['Team']
    probabilities_per_team = dict(zip(teams, probabilities[:, 1]))
        
    top_4_teams = heapq.nlargest(4, probabilities_per_team.items(), key=lambda x: x[1]) # get the top 4 teams based on probabilities
        
    real_winning_team = None
    if not is_actual:
        real_winning_team = new_statistics[new_statistics['StanleyCup'] == 1]['Team'].values[0]
        
    predicted_winning_team = top_4_teams[0][0]

    return real_winning_team, predicted_winning_team, top_4_teams

def print_results(file, real_winning_team, predicted_winning_team, prediction_correct, top_4_teams):
    # This function prints the results for a given file
    print(f"For file {file}:")
    if not is_actual:
        print(f"    The real Stanley Cup winning team is: {real_winning_team}")
    print(f"    The predicted Stanley Cup winning team is: {predicted_winning_team}")
    print(f"    The four highest probabilities teams are: {', '.join([f'{format(prob * 100, '.2f')}% {team}' for team, prob in top_4_teams])}") # multiply the probabilities by 100 and use .4f instead of .2f to keep four decimal places
    if not is_actual:
        print(f"    Prediction is {'correct' if prediction_correct else 'incorrect'} ({total_correct_predictions} / {total_predictions}) \n")
    else:
        print("    We will know at the end of the season if prediction is correct")

for file in os.listdir(directory):
    if file.endswith('.xlsx') and not file.__contains__('18_last_seasons'):
        
        is_actual = False
        if file.__contains__('ACTUAL'):
            is_actual = True

        real_winning_team, predicted_winning_team, top_4_teams = process_file(file)
        
        if not is_actual:
            prediction_correct = predicted_winning_team == real_winning_team
            total_predictions += 1
            if prediction_correct:
                total_correct_predictions += 1

        print_results(file, real_winning_team, predicted_winning_team, prediction_correct, top_4_teams)
