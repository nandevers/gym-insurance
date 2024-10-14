from gym.envs.registration import register

# Register the custom environment
register(
    id="InsurEnv-v0",  # Unique identifier for the environment
    entry_point="gym_insurance.envs.insurenv_v2:InsurEnv",  # Path to the InsurEnv class
    max_episode_steps=1000,  # Set maximum number of steps per episode
    reward_threshold=200.0,  # Optional reward threshold for considering the environment solved
    nondeterministic=False,  # Set to True if the environment is not deterministic
    kwargs={
        "data": None,  # Placeholder for actual data input when initializing the environment
        "index_column": "case_id",  # Example of index column (to be adjusted based on actual data)
        "value_column": "claim_value",  # Example of value column (to be adjusted based on actual data)
        "state_columns": ["feature_1", "feature_2", "feature_3"],  # Example state columns (to be adjusted)
        "budget": 10000.0,  # Example initial budget
    },
)
