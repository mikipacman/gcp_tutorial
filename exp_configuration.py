from mrunner.helpers.specification_helper import create_experiments_helper
import os

assert 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ
assert 'NEPTUNE_API_TOKEN' in os.environ
assert 'NEPTUNE_PROJECT_NAME' in os.environ

model_params = {
    "learning_rate": 1e-3,
    "prioritized_replay": True,
    "verbose": 1,
}

learn_params = {
    "total_timesteps": int(2e4),
    "log_interval": 1,
}

base_config = {
    **model_params,
    **learn_params,
}

params_grid = {
    "learning_rate": [3e-4, 4e-3],
}

experiments_list = create_experiments_helper(experiment_name='lunar lander mrunner test',
                                             base_config=base_config,
                                             params_grid=params_grid,
                                             script=f'python /gcp_tutorial/train_with_mrunner.py',
                                             exclude=[],
                                             python_path='',
                                             tags=['some', 'tags'],
                                             with_neptune=True,
                                             env={"GOOGLE_APPLICATION_CREDENTIALS": "google_app_cred.json"})
