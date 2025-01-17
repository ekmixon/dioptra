# This Software (Dioptra) is being made available as a public service by the
# National Institute of Standards and Technology (NIST), an Agency of the United
# States Department of Commerce. This software was developed in part by employees of
# NIST and in part by NIST contractors. Copyright in portions of this software that
# were developed by NIST contractors has been licensed or assigned to NIST. Pursuant
# to Title 17 United States Code Section 105, works of NIST employees are not
# subject to copyright protection in the United States. However, NIST may hold
# international copyright in software created by its employees and domestic
# copyright (or licensing rights) in portions of software that were assigned or
# licensed to NIST. To the extent that NIST holds copyright in this software, it is
# being made available under the Creative Commons Attribution 4.0 International
# license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts
# of the software developed or licensed by NIST.
#
# ACCESS THE FULL CC BY 4.0 LICENSE HERE:
# https://creativecommons.org/licenses/by/4.0/legalcode
"""A task plugin module for managing random number generators."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import structlog
from numpy.random._generator import Generator as RNGenerator
from structlog.stdlib import BoundLogger

from mitre.securingai import pyplugs

LOGGER: BoundLogger = structlog.stdlib.get_logger()


@pyplugs.register
@pyplugs.task_nout(2)
def init_rng(seed: int = -1) -> Tuple[int, RNGenerator]:
    """Constructs a new random number generator.

    Args:
        seed: A seed to initialize the random number generator. If the value is less
            than zero, then the seed is generated by pulling fresh, unpredictable
            entropy from the OS. The default is `-1`.

    Returns:
        A tuple containing the seed and the initialized random number generator. If a
        `seed < 0` was passed as an argument, then the seed generated by the OS will be
        returned.

    See Also:
        - :py:func:`numpy.random.default_rng`
    """
    rng = np.random.default_rng(seed if seed >= 0 else None)

    if seed < 0:
        seed = rng.bit_generator._seed_seq.entropy

    return int(seed), rng
