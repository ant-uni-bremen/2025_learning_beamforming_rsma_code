
from pathlib import Path
from shutil import rmtree
import gzip
import pickle

import tensorflow as tf

import src


def save_model_checkpoint(
        config: 'src.config.config.Config',
        networks: list,
        norm_dict: dict,
        logger,
        high_scores: list,
        parent_string: str,
        extra_string: str = None,
) -> (Path, list):

    name = f''
    if extra_string is not None:
        name += extra_string + '_'
    name += f'snap_{high_scores[-1]:.3f}'

    checkpoint_path = Path(
        config.trained_models_path,
        config.config_learner.training_name,
        parent_string,
        name,
    )

    if len(networks) == 1:
        tf.saved_model.save(networks[0], str(Path(checkpoint_path, 'model')))
    else:
        for network_id, network in enumerate(networks):
            sac_path = Path(checkpoint_path, f'agent_{network_id}')
            tf.saved_model.save(network, str(Path(sac_path, 'model')))

    logger.info(f'Saved model checkpoint at mean reward {high_scores[-1]:.3f}')

    # save config
    config.save(Path(checkpoint_path, 'config'))

    # save norm dict
    with gzip.open(Path(checkpoint_path, 'config', 'norm_dict.gzip'), 'wb') as file:
        pickle.dump(norm_dict, file)

    # clean model checkpoints
    for high_score_prior_id, high_score_prior in enumerate(reversed(high_scores)):
        if high_scores[-1] > 1.05 * high_score_prior or high_score_prior_id >=3:
            if extra_string is not None:
                name = f'{extra_string}_snap_{high_score_prior:.3f}'
            else:
                name = f'snap_{high_score_prior:.3f}'

            prior_checkpoint_path = Path(
                config.trained_models_path,
                config.config_learner.training_name,
                parent_string,
                name
            )
            rmtree(path=prior_checkpoint_path, ignore_errors=True)
            high_scores.remove(high_score_prior)

    return checkpoint_path, high_scores