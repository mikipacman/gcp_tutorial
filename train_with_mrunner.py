import gym
import neptune
from stable_baselines import DQN
from stable_baselines.logger import KVWriter
from stable_baselines.logger import Logger
from mrunner.helpers.client_helper import get_configuration


class NeptuneFormat(KVWriter):
	def writekvs(self, kvs):
		for key, value in sorted(kvs.items()):
			neptune.send_metric(key, value)


if __name__ == "__main__":
	params = get_configuration(with_neptune=True, inject_parameters_to_gin=True)
	Logger.CURRENT = Logger(folder=None, output_formats=[NeptuneFormat()])

	# Create environment
	env = gym.make('LunarLander-v2')

	# Instantiate the agent
	model = DQN('MlpPolicy',
				env,
				learning_rate=params["learning_rate"],
				prioritized_replay=params["prioritized_replay"],
				verbose=params["verbose"])

	# Train the agent
	model.learn(total_timesteps=params["total_timesteps"],
				log_interval=params["log_interval"])

	# Save the agent
	model.save("/log/dqn_lunar")
