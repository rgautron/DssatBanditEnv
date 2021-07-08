from gym.envs.registration import register

register(
    id='dssat-v0',
    entry_point='gym_dssat.envs:DssatEnv',
)