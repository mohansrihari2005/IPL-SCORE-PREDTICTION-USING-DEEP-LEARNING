import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras # Neural Networks library on the Top of TensorFlow
import tensorflow as tf # Neurall Networks Library

ipl = pd.read_csv('ipl_data.csv') # EDA: 
ipl.head()
#Dropping certain features 
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis =1)
#Further Pre-Processing
X = df.drop(['total'], axis =1) # Set of features
y = df['total'] # Target or output or label which we want to predict
#X.describe()
X.info()
#Label Encoding
#Label Encoding
from sklearn.preprocessing import LabelEncoder
# Create a LabelEncoder object for each categorical feature
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()
# Fit and transform the categorical features with label encoding
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])
X
#Train Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train
#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled
#Defining neural network
# Define the neural network model
model = keras.Sequential([
    keras.layers.Input( shape=(X_train_scaled.shape[1],)),  # Input layer
    keras.layers.Dense(512, activation='relu'),  # Hidden layer with 512 units and ReLU activation
    keras.layers.Dense(216, activation='relu'),  # Hidden layer with 216 units and ReLU activation
    keras.layers.Dense(1, activation='linear')  # Output layer with linear activation for regression
])
# Compile the model with Huber loss
huber_loss = tf.keras.losses.Huber(delta=1.0)  # You can adjust the 'delta' parameter as needed
model.compile(optimizer='adam', loss=huber_loss)  # Use Huber loss for regression
#Model Training
# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))
#After the training, we have stored the training and validation loss values to our neural network during the training process.
model_losses = pd.DataFrame(model.history.history)
model_losses.plot()
#Model Evaluation

# Make predictions
predictions = model.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(y_test,predictions)
# Let’s create an Interactive Widget

import ipywidgets as widgets
from IPython.display import display, clear_output

import warnings
warnings.filterwarnings("ignore")

venue = widgets.Dropdown(options=df['venue'].unique().tolist(),description='Select Venue:')
batting_team = widgets.Dropdown(options =df['bat_team'].unique().tolist(),  description='Select Batting Team:')
bowling_team = widgets.Dropdown(options=df['bowl_team'].unique().tolist(),  description='Select Bowling Team:')
striker = widgets.Dropdown(options=df['batsman'].unique().tolist(), description='Select Striker:')
bowler = widgets.Dropdown(options=df['bowler'].unique().tolist(), description='Select Bowler:')

predict_button = widgets.Button(description="Predict Score")

def predict_score(b):
    with output:
        clear_output()  # Clear the previous output
        

        # Decode the encoded values back to their original values
        decoded_venue = venue_encoder.transform([venue.value])
        decoded_batting_team = batting_team_encoder.transform([batting_team.value])
        decoded_bowling_team = bowling_team_encoder.transform([bowling_team.value])
        decoded_striker = striker_encoder.transform([striker.value])
        decoded_bowler = bowler_encoder.transform([bowler.value])


        input = np.array([decoded_venue,  decoded_batting_team, decoded_bowling_team,decoded_striker, decoded_bowler])
        input = input.reshape(1,5)
        input = scaler.transform(input)
        #print(input)
        predicted_score = model.predict(input)
        predicted_score = int(predicted_score[0,0])

        print(predicted_score)

#The widget-based interface allows you to interactively predict the score for specific match scenarios. 
#Now, we have set up the button to trigger the predict_score function when clicked and display the widgets for venue, batting team , bowling team, striker and bowler.

predict_button.on_click(predict_score)
output = widgets.Output()
display(venue, batting_team, bowling_team, striker, bowler, predict_button, output)

#EDA for the data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ipl_data.csv')
df.head()

#1.Get an Overview of the Data
#Shape and Basic Info: First, check the shape of the dataset (number of rows and columns), and get basic info on data types, missing values, and sample data.
df.info()
#2. Summary Statistics
#For numerical features (e.g., runs scored, wickets, overs, etc.), generate summary statistics to understand the distribution and central tendency.
#This will give you details like:
#Average or arithmetic mean  = mean_value = statistics.mean(data),
#median_value = statistics.median(data),
#population_std_dev = statistics.pstdev(data)
#Min, Max values
#Quartiles (25%, 50%, 75%)=df['values'].quantile([0.25, 0.5, 0.75]).......#we should import pandas for this
df.describe()

#3. Check for Missing Values
# Check for missing values
df.isnull().sum()

#4. Examine Categorical Variables


#For categorical variables like "Batting Team", "Bowler", "Venue", you can check the distribution of each category:

# Check distribution of categorical features
print(df['venue'].value_counts())
print(df['bat_team'].value_counts())
print(df['bowl_team'].value_counts())

#5.Group by 'venue' and count the number of unique match_ids for each venue
matches_per_venue = df.groupby('venue')['mid'].nunique()

# Display the result
print(matches_per_venue)
#bar graph representation of above data

# Count number of unique matches played at each venue
matches_per_venue = df.groupby('venue')['mid'].nunique()

# Plot the bar graph with 'hue' and suppress the legend
plt.figure(figsize=(10, 6))
sns.barplot(x=matches_per_venue.index, y=matches_per_venue.values, palette="Blues_d", hue=matches_per_venue.index, legend=False)
plt.title('Number of Matches Played at Each Venue')
plt.xlabel('Venue')
plt.ylabel('Number of Matches')
plt.xticks(rotation=90)  # Rotate x labels for better readability
plt.show()

#6.no.of.matches a particular team has played
# Group by 'venue' and count the number of unique match_ids for each venue
no_of_matches= df.groupby('bat_team')['mid'].nunique()
# Display the result
print(no_of_matches)
#7.Count unique teams in the 'bat_team' column
unique_batting_teams = df['bat_team'].unique()
print("Unique Batting Teams:", unique_batting_teams)
print("Number of total Batting Teams:", len(unique_batting_teams))

# Count unique teams in the 'bowl_team' column
unique_bowling_teams = df['bowl_team'].unique()
print("Unique Bowling Teams:", unique_bowling_teams)
print("Number of total Bowling Teams:", len(unique_bowling_teams))

#8.Count unique batsmen in the dataset
unique_batsmen = df['batsman'].unique()
#print("Unique Batsmen:", unique_batsmen)
print("Number of Unique Batsmen:", len(unique_batsmen))

#10.Count unique bowlers in the dataset
unique_bowlers = df['bowler'].unique()
#print("Unique Bowlers:", unique_bowlers)
print("Number of Unique Bowlers:", len(unique_bowlers))

#11. Runs Scored by Each Team (Batting Team)

import matplotlib.pyplot as plt
import seaborn as sns

# Group by 'bat_team' and sum the runs scored by each team
runs_per_batting_team = df.groupby('bat_team')['runs'].sum()

# Plot the bar graph
plt.figure(figsize=(10, 6))

# Assign 'bat_team' to 'x' and the runs values to 'y', and use 'hue' for 'bat_team' to avoid the FutureWarning
sns.barplot(x=runs_per_batting_team.index, y=runs_per_batting_team.values, palette="coolwarm", hue=runs_per_batting_team.index, legend=False)

plt.title('Total Runs Scored by Each Batting Team')
plt.xlabel('Batting Team')
plt.ylabel('Total Runs')
plt.xticks(rotation=90)
plt.show()
#wickets taken by each team
# Group by 'bowl_team' and sum the wickets taken by each team
wickets_per_bowling_team = df.groupby('bowl_team')['wickets'].sum()

# Plot the bar graph with 'hue' explicitly defined
plt.figure(figsize=(10, 6))
sns.barplot(x=wickets_per_bowling_team.index, 
            y=wickets_per_bowling_team.values, 
            palette="magma", 
            hue=wickets_per_bowling_team.index)  # Assigning hue to x-axis categories
plt.title('Total Wickets Taken by Each Bowling Team')
plt.xlabel('Bowling Team')
plt.ylabel('Total Wickets')
plt.xticks(rotation=90)  # Rotate x labels for better readability
plt.show()
#Top 10 Batsmen with Most Runs

# Group by 'batsman' and sum the runs scored by each batsman
runs_per_batsman = df.groupby('batsman')['runs'].sum()

# Sort the batsmen by runs and select top 10
top_10_batsmen = runs_per_batsman.sort_values(ascending=False).head(10)

# Plot the bar graph with 'hue' explicitly defined
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_batsmen.index, 
            y=top_10_batsmen.values, 
            palette="YlGnBu", 
            hue=top_10_batsmen.index)  # Assigning hue to x-axis categories (batsman names)
plt.title('Top 10 Batsmen with Most Runs')
plt.xlabel('Batsman')
plt.ylabel('Total Runs')
plt.xticks(rotation=90)  # Rotate x labels for better readability
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Group by 'bowler' and sum the wickets taken by each bowler
wickets_per_bowler = df.groupby('bowler')['wickets'].sum()

# Sort the bowlers by wickets and select top 10
top_10_bowlers = wickets_per_bowler.sort_values(ascending=False).head(10)

# Plot the bar graph with 'hue' explicitly defined
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_bowlers.index, 
            y=top_10_bowlers.values, 
            palette="magma", 
            hue=top_10_bowlers.index)  # Assigning hue to x-axis categories (bowler names)
plt.title('Top 10 Bowlers with Most Wickets')
plt.xlabel('Bowler')
plt.ylabel('Total Wickets')
plt.xticks(rotation=90)  # Rotate x labels for better rea

#runs scored by a specific batsman against a particular bowler
# Group by 'batsman' and 'bowler', then sum the runs scored by each batsman against each bowler
runs_against_bowler = df.groupby(['batsman', 'bowler'])['runs'].sum().reset_index()

# Specify the batsman and bowler you're interested in
batsman_name = 'SC Ganguly'  # Replace with the desired batsman's name
bowler_name = 'Z Khan'   # Replace with the desired bowler's name

# Filter the data for the specific batsman and bowler
runs_for_batsman_against_bowler = runs_against_bowler[(runs_against_bowler['batsman'] == batsman_name) & 
                                                     (runs_against_bowler['bowler'] == bowler_name)]
# Check if the filtered data exists
if not runs_for_batsman_against_bowler.empty:
    total_runs = runs_for_batsman_against_bowler['runs'].sum()  # Get the total runs
    print(f'Total runs scored by {batsman_name} against {bowler_name}: {total_runs}')
else:
    print(f"No data available for {batsman_name} against {bowler_name}")




