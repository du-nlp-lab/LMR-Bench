# Copyright 2023 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall

from hierarchy_transformers.models import HierarchyTransformer
from hierarchy_transformers.utils import format_citation

logger = logging.getLogger(__name__)


class HierarchyTransformerLoss(torch.nn.Module):
    """Hyperbolic loss that linearly combines hperbolic clustering loss and hyperbolic Centripetal loss and applies weights for joint optimisation."""

    def __init__(
        self,
        model: HierarchyTransformer,
        clustering_loss_weight: float = 1.0,
        clustering_loss_margin: float = 5.0,
        centripetal_loss_weight: float = 1.0,
        centripetal_loss_margin: float = 0.5,
    ):
        super().__init__()

        self.model = model
        self.manifold = self.model.manifold
        self.cluster_weight = clustering_loss_weight
        self.centri_weight = centripetal_loss_weight
        self.cluster_loss_margin = clustering_loss_margin
        self.centri_loss_margin = centripetal_loss_margin

    def forward(self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor):
        """Forward propagation that follows [`sentence_transformers.losses`](https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers/losses)."""
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 3
        rep_anchor, rep_positive, rep_negative = reps

        combined_loss, cluster_loss, centri_loss = None, None, None
        
        ### TODO: compute and combine hyperbolic clustering and centripetal losses here

        return {
            "loss": combined_loss,
            "cluster_loss": cluster_loss,
            "centri_loss": centri_loss,
        }

    @property
    def citation(self) -> str:
        return format_citation(
            """
            @article{he2024language,
              title={Language models as hierarchy encoders},
              author={He, Yuan and Yuan, Zhangdie and Chen, Jiaoyan and Horrocks, Ian},
              journal={arXiv preprint arXiv:2401.11374},
              year={2024}
            }
            """
        )


class HyperbolicClusteringLoss(torch.nn.Module):
    r"""Hyperbolic loss that clusters entities in subsumptions.

    Essentially, this loss is expected to achieve:

    $$d(child, parent) < d(child, negative)$$

    Inputs are presented in `(rep_anchor, rep_positive, rep_negative)`.
    """

    def __init__(self, manifold: PoincareBall, margin: float):
        super().__init__()
        self.manifold = manifold
        self.margin = margin

    def get_config_dict(self):
        config = {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist",
            "margin": self.margin,
        }
        return config

    def forward(self, rep_anchor: torch.Tensor, rep_positive: torch.Tensor, rep_negative: torch.Tensor):
        pass

    @property
    def citation(self) -> str:
        return format_citation(
            """
            @article{he2024language,
              title={Language models as hierarchy encoders},
              author={He, Yuan and Yuan, Zhangdie and Chen, Jiaoyan and Horrocks, Ian},
              journal={arXiv preprint arXiv:2401.11374},
              year={2024}
            }
            """
        )


class HyperbolicCentripetalLoss(torch.nn.Module):
    r"""Hyperbolic loss that regulates the norms of child and parent entities.

    Essentially, this loss is expected to achieve:

    $$d(child, origin) > d(parent, origin)$$

    Inputs are presented in `(rep_anchor, rep_positive, rep_negative)` but only `(rep_anchor, rep_positive)` pairs are involved in this loss.
    """

    def __init__(self, manifold: PoincareBall, margin: float):
        super().__init__()
        self.manifold = manifold
        self.margin = margin

    def get_config_dict(self):
        config = {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist0",
            "margin": self.margin,
        }
        return config

    def forward(self, rep_anchor: torch.Tensor, rep_positive: torch.Tensor, rep_negative: torch.Tensor):
        pass
 
    @property
    def citation(self) -> str:
        return format_citation(
            """
            @article{he2024language,
              title={Language models as hierarchy encoders},
              author={He, Yuan and Yuan, Zhangdie and Chen, Jiaoyan and Horrocks, Ian},
              journal={arXiv preprint arXiv:2401.11374},
              year={2024}
            }
            """
        )
