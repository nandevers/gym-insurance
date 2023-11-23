import pandas as pd
from constants import DATAPATH, logger

def read_dqn_results():
    logger.info("Reading DQN results")
    dqn_results = pd.read_parquet(DATAPATH / "dqn")
    dqn_results["episodes"] = dqn_results["episodes"].astype(int)-2
    dqn_results["experiment"] = "dqn"
    return dqn_results 

def read_random_results():
    logger.info("Reading random results")
    random_results = pd.read_parquet(DATAPATH / "random")
    random_results["episodes"] = random_results["episodes"].astype(int)
    random_results["episodes"] = random_results["episodes"] - 3
    random_results.drop_duplicates(inplace=True)
    random_results["experiment"] = "random"
    return random_results

def read_all_results():
    logger.info("Joining Data")
    return pd.concat([read_random_results(), read_dqn_results()], axis = 0).drop_duplicates()

if __name__=="__main__":
    final_data = read_all_results()
    final_data["decision"] = final_data['decision'].astype(int)
    final_data["episodes"] = final_data["episodes"].astype(int)
    final_data.loc[:, ["experiment", "episodes", "steps", "wc", "woc", "rewards", "decision","valor_indenizacao", 'first_digit_0', 'first_digit_1', 'first_digit_2', 'first_digit_3',
       'first_digit_4', 'first_digit_5', 'first_digit_6', 'first_digit_7',
       'first_digit_8', 'first_digit_9', 'first_digit_10', 'second_digit_0',
       'second_digit_1', 'second_digit_2', 'second_digit_3', 'second_digit_4',
       'second_digit_5', 'second_digit_6', 'second_digit_7', 'second_digit_8',
       'second_digit_9', 'second_digit_10', 'second_digit_11', 'third_digit_0',
       'third_digit_1', 'third_digit_2', 'third_digit_3', 'third_digit_4',
       'third_digit_5', 'third_digit_6', 'third_digit_7', 'third_digit_8',
       'third_digit_9', 'third_digit_10', 'third_digit_11']].to_csv("my_rewards.csv")
    logger.info("Writing final dataset.")
    final_data.to_csv(DATAPATH / 'final_results.csv')