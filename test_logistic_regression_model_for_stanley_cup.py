import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

model = joblib.load('logistic_regression_model_for_stanley_cup.pkl')

directory = 'C:\\Users\\Jeremy\\Dev\\Python\\logistic_regression_model_for_stanley_cup\\nhl_data'

total_predictions = 0
total_correct_predictions = 0

for file in os.listdir(directory):
    if file.endswith('.xlsx'):
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
        
        top_16_teams = sorted(probabilities_per_team.items(), key=lambda x: x[1], reverse=True)[:16]
        
        #real_winning_team = new_statistics[new_statistics['StanleyCup'] == 1]['Team'].values[0]
        
        predicted_winning_team = top_16_teams[0][0]
        #prediction_correct = predicted_winning_team == real_winning_team
        
        #total_predictions += 1

        #if prediction_correct:
        #    total_correct_predictions += 1

        print(f"For file {file}:")
        #print(f"    The real Stanley Cup winning team is: {real_winning_team}")
        print(f"    The predicted Stanley Cup winning team is: {predicted_winning_team}")
        #print(f"    Prediction is {'correct' if prediction_correct else 'incorrect'} ({total_correct_predictions} / {total_predictions}) \n")