from mushroom_rl.core import Core, Agent
from loco_mujoco import LocoEnv

log_path = "./logs/loco_mujoco_evalution_2024-09-29_18-25-54/env_id___HumanoidTorque.walk/2/agent_epoch_189_J_92.139050.msh"

env = LocoEnv.make("HumanoidTorque.walk")
agent = Agent.load(log_path)

core = Core(agent, env)

core.evaluate(n_episodes=10, render=True, record=True)