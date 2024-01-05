from rllib_env_warpper import RllibAnalogDesignAutoEnv
from ray.rllib.models import ModelCatalog

config = {
    "env": RllibAnalogDesignAutoEnv,
}

preprocessors = ModelCatalog.get_preprocessor(
    "PPO", env_config={"observation_space": config["env"].observation_space_sample})

for preprocessor in preprocessors:
    print(preprocessor.__class__.__name__, preprocessor.config)