"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import sys

import imageio
import numpy as np
import tensorflow as tf
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def create_policy(ckpt) -> _policy.Policy:
    config="pi0_libero"
    dir=ckpt
    return _policy_config.create_trained_policy(
            _config.get_config(config), dir)