#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("dockerfile_name", ["Dockerfile", "Dockerfile.deploy"])
def test_python_dependencies_are_built_with_clang(dockerfile_name):
    dockerfile = (REPO_ROOT / dockerfile_name).read_text()
    builder = dockerfile.split("FROM base AS builder", maxsplit=1)[1].split("FROM base AS production", maxsplit=1)[0]

    assert "apt install -y build-essential clang " in builder
    assert "CC=clang CXX=clang++ uv sync --python 3.13" in builder
