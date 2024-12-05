# Systems and Tool-Chains for AI Engineers
## Analysis of the FIFA Complete Player Dataset
https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset

### Contributers
 - Jayant Sharma (jayantsh)
 - Rishi Mandyam (rmandyam)

 This project describes the data cleaning, data engineering, model set up and tuning from scratch for four different machine learning models, on a dataset of FIFA players over the years 2015-2022, to predict the overall attributes and game results. Packages and technologies such as SparkML, PyTorch, Apache Hadoop, PostgreSQL, Apache Kafka, Docker, Kubernetes, GitHub Actions etc have been used for the implementation. 

 The models were run locally, as well as on the Google Cloud Platform. The model was deployed using containerization though Docker, and the outcomes were streamed across different consumers through Apache Kafka. 

### Feature Descriptions
<details>
  <summary><strong>Player Information</strong></summary>
  
  - **sofifa_id** - Unique identifier for a player in the FIFA dataset.
  - **player_url** - URL of the player’s profile on the official FIFA website.
  - **short_name** - Player's abbreviated or common name.
  - **long_name** - Player's full official name.
  - **player_positions** - Positions on the field that the player can play.
  - **overall** - Player's overall rating, reflecting current ability.
  - **potential** - Maximum potential rating the player can achieve.
  - **age** - Player's current age.
  - **dob** - Player’s date of birth.
  - **height_cm** - Player's height in centimeters.
  - **weight_kg** - Player's weight in kilograms.
</details>

<details>
  <summary><strong>Club Information</strong></summary>

  - **club_team_id** - Unique identifier for the club team the player is currently playing for.
  - **club_name** - Name of the club the player is affiliated with.
  - **league_name** - Name of the league where the player’s club competes.
  - **league_level** - Tier or level of the league (e.g., 1 for top-tier).
  - **club_position** - Player’s position on the field in the club.
  - **club_jersey_number** - Player’s jersey number in the club team.
  - **club_loaned_from** - If on loan, the name of the club the player is loaned from.
  - **club_joined** - The date when the player joined the current club.
  - **club_contract_valid_until** - The year when the player’s club contract expires.
</details>

<details>
  <summary><strong>Nationality Information</strong></summary>

  - **nationality_id** - Unique identifier for the player’s nationality.
  - **nationality_name** - Name of the player’s nationality.
  - **nation_team_id** - Unique identifier for the national team the player represents.
  - **nation_position** - Player’s position in the national team.
  - **nation_jersey_number** - Player’s jersey number in the national team.
</details>

<details>
  <summary><strong>Physical Attributes</strong></summary>

  - **body_type** - Player’s body type (e.g., stocky, lean).
  - **real_face** - Whether the player has a real face model in the game (True/False).
  - **release_clause_eur** - The release clause for the player in euros.
  - **preferred_foot** - The player's dominant foot (left or right).
  - **weak_foot** - Player's weak foot rating (out of 5).
  - **skill_moves** - Player’s skill moves rating (out of 5).
</details>

<details>
  <summary><strong>Skill and Performance Ratings</strong></summary>

  - **pace** - Overall rating of the player's speed and acceleration.
  - **shooting** - Overall rating of the player's ability to shoot.
  - **passing** - Overall rating of the player's passing ability.
  - **dribbling** - Overall rating of the player's dribbling ability.
  - **defending** - Overall rating of the player's defending ability.
  - **physic** - Overall rating of the player's physical attributes.
</details>

<details>
  <summary><strong>Attacking Attributes</strong></summary>

  - **attacking_crossing** - Rating of the player's ability to cross the ball.
  - **attacking_finishing** - Rating of the player's ability to finish scoring chances.
  - **attacking_heading_accuracy** - Rating of the player's heading accuracy.
  - **attacking_short_passing** - Rating of the player's short passing ability.
  - **attacking_volleys** - Rating of the player's volleying ability.
</details>

<details>
  <summary><strong>Skill Attributes</strong></summary>

  - **skill_dribbling** - Rating of the player’s dribbling skills.
  - **skill_curve** - Rating of the player’s ability to curve the ball.
  - **skill_fk_accuracy** - Rating of the player’s accuracy in free kicks.
  - **skill_long_passing** - Rating of the player’s long passing ability.
  - **skill_ball_control** - Rating of the player’s control over the ball.
</details>

<details>
  <summary><strong>Movement Attributes</strong></summary>

  - **movement_acceleration** - Rating of the player's acceleration speed.
  - **movement_sprint_speed** - Rating of the player’s sprint speed.
  - **movement_agility** - Rating of the player’s agility.
  - **movement_reactions** - Rating of the player's reaction time.
  - **movement_balance** - Rating of the player's balance while moving.
</details>

<details>
  <summary><strong>Power Attributes</strong></summary>

  - **power_shot_power** - Rating of the player’s shot power.
  - **power_jumping** - Rating of the player’s jumping ability.
  - **power_stamina** - Rating of the player’s stamina.
  - **power_strength** - Rating of the player’s strength.
  - **power_long_shots** - Rating of the player's ability to take long-range shots.
</details>

<details>
  <summary><strong>Mentality Attributes</strong></summary>

  - **mentality_aggression** - Rating of the player's aggression on the field.
  - **mentality_interceptions** - Rating of the player's ability to intercept the ball.
  - **mentality_positioning** - Rating of the player's positioning in attack.
  - **mentality_vision** - Rating of the player's ability to see and execute passes.
  - **mentality_penalties** - Rating of the player’s ability to take penalties.
  - **mentality_composure** - Rating of the player’s composure under pressure.
</details>

<details>
  <summary><strong>Defending Attributes</strong></summary>

  - **defending_marking_awareness** - Rating of the player's marking ability and defensive awareness.
  - **defending_standing_tackle** - Rating of the player’s standing tackle ability.
  - **defending_sliding_tackle** - Rating of the player’s sliding tackle ability.
</details>

<details>
  <summary><strong>Goalkeeping Attributes</strong></summary>

  - **goalkeeping_diving** - Rating of the goalkeeper's diving ability.
  - **goalkeeping_handling** - Rating of the goalkeeper's handling of the ball.
  - **goalkeeping_kicking** - Rating of the goalkeeper's kicking ability.
  - **goalkeeping_positioning** - Rating of the goalkeeper's positioning.
  - **goalkeeping_reflexes** - Rating of the goalkeeper's reflexes.
  - **goalkeeping_speed** - Rating of the goalkeeper's speed.
</details>

<details>
  <summary><strong>Player Images and Logos</strong></summary>

  - **player_face_url** - URL to the player's face image.
  - **club_logo_url** - URL to the club’s logo.
  - **club_flag_url** - URL to the club’s flag.
  - **nation_logo_url** - URL to the national team’s logo.
  - **nation_flag_url** - URL to the national team’s flag.
</details>


### Steps to run code
1. Make sure the username and password of the PostgreSQL properties cell is the one where you want to ingest the data. Since the mode is 'overwrite' and there is only dataframe ingested, there would be no problem. 
2. Ensure that the counts of the entire dataset is 144323 in your PostgreSQL PGAdmin Shell.
3. There are a lot of NULL count calculation cells, which can be ignored when the code is being run.
4. To run the hyperparameter tuning with Cross Validation on the SparkML Models, uncomment the cells specified and use the hyperparamters output by the bestModel in the Linear Regressor and Random Forest Regressor in the next respective cells. 

### Model and Hyperparameter Choices
- **Apache Spark**
- Random Forest: This model was chosen because of its ability to handle high-dimensional data and complex, nonlinear relationships effectively. It combines predictions from multiple decision trees, which reduces overfitting. This approach helps the model generalize well on the test data, resulting in more accurate predictions. The hyper parameters we tuned for this model were the number of trees and the maximum depth of each tree as well as the maximum number of bins to divide the feature values at each node in the decision trees. We tuned the number of trees to find a balance between accuracy and computation time and we tuned the max depth and maxBins as to maximize accuracy without overfitting on the test data.

numTrees: Helps to average out noise and avoid overfitting by learning from different subsets of features and data samples.
maxDepth: Provides a balance between complexity and generalizability by controlling how deep individual trees go, which can capture the varied relationships between player attributes and their overall score.
maxBins: Enhances the ability to split continuous features meaningfully, which is beneficial when predicting scores influenced by subtle variations in continuous player attributes.

Metrics Used: For Random Forest Hyperparameter tuning, R2 square was used as it tells whether the model is capturing the variance correctly. Since our model has very long range dependent variables, we wanted to try it out for one model. 

Mean Squared Error and R squared were chosen to test the performance of the model on the test dataset. The reason we did not use Accuracy is that it does not make sense to report it for a regressor, as the predictions might never be equal to the target (as there are no fixed classes), but how far they are from the target is necessary (MSE).
  
- Linear Regression: Although this model is simple, it trains data quickly and is computationally efficient. Similarly to the random forest regressor, this model generalizes well on test data. The hyperparameters we decided to tune were the maximum number of iterations as well as the regularization parameter. The maxIter parameter represents the maximum number of iterations the optimizer will step to reach the solution. More steps may lead to a higher accuracy but this can increase the training time. The second hyperparameter that was tuned was Reg-Param which helps prevent overfitting by adding a penalty to large coefficients in the model but it can encourage underfitting if the Reg-Param value is too small. As a result, this hyperparameter was chosen to tune such that accuracy was maximised and overfitting was minimized.

Similar as RF, used MSE for hyperparameter tuning and to test the performance, we used MSE and R2. (reason is same as above)
  
- **Pytorch**
- Multi-Layer-Perceptron: In this multi layer perceptron we used 4 hidden layers with 128 nodes in each layer. We chose that structure because it was a deep neural network that captures the complexities of the dataset efficiently. The hyperparameters we chose to tune for this model were the learning rate and number of epochs. The learning rate was decreased from 0.05 to 0.005 so that the model achieves a more stable solution without overshooting, and can converge well slowly as the number of epochs were high as well. The number of epochs were increased from 10 to 20 to give the model more runs to decrease the MSE loss. 
  
- Linear Regression: This model was again chosen because it is computationally efficient, trains quickly, and generalizes well on test data. We tuned two key hyperparameters: the optimizer and the learning rate. Switching the optimizer to ADAM helps the model converge faster and handles noisy gradients effectively, improving performance because of the additional momentum attribute. Adjusting the learning rate allowed us to control the step size of each iteration—balancing faster training with stability. By fine-tuning the learning rate, we aimed to achieve optimal accuracy without overfitting.

Metrics used: Mean Squared Error and R squared are used as metrics to test the performance, as for a regressor these are the best metrics to use as there are no classes, capturing how 'far' the predictions are from the target is necessary, that MSE does. R square tells how the model captures the variance, which we felt was important. 


**Image Files:**
The two png file in this repository shows the Random Forest Model's and the Linear Regressor's best hyperparameters after running it on the cloud and the Jupyter Notebook respectively. Since it was computationally very expensive, we chose to comment it out in the main file.
