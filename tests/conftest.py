###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import pytest
import numpy as np

import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from tests.input_data import InputData


@pytest.fixture
def sample_input_data():
    input_data = InputData()
    return input_data.data_for_pipeline()
