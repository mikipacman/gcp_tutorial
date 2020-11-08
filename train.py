import gym
import neptune
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.logger import KVWriter
from stable_baselines.logger import Logger


class NeptuneFormat(KVWriter):
	def writekvs(self, kvs):
		for key, value in sorted(kvs.items()):
			neptune.send_metric(key, value)


model_params = {
	"learning_rate": 1e-3,
	"prioritized_replay": True,
	"verbose": 1,
}

learn_params = {
	"total_timesteps": int(2e4),
	"log_interval": 1,
}

all_params = {
	**model_params,
	**learn_params,
}


if __name__ == "__main__":

	neptune.init()
	with neptune.create_experiment(name="lunar lander test", params=all_params):
		Logger.CURRENT = Logger(folder=None, output_formats=[NeptuneFormat()])
		# Create environment
		env = gym.make('LunarLander-v2')

		# Instantiate the agent
		model = DQN('MlpPolicy', env, **model_params)
		# Train the agent
		model.learn(**learn_params)

		# Evaluate the agent
		mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

		# Enjoy trained agent
		obs = env.reset()
		for i in range(1000):
			action, _states = model.predict(obs)
			obs, rewards, dones, info = env.step(action)
			env.render()
