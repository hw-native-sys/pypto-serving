# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Parity harness for the DeepSeekV4 staged weight pipeline (#163 step 1).
#
# The refactor must keep the stacked slabs byte-identical, so before any of it happens
# these tests establish that a byte-level fingerprint of the real load -> pack -> stack
# path is (a) reproducible and (b) actually sensitive to the properties the refactor could
# break. A parity test that cannot fail is worse than none: it licenses the change it was
# supposed to guard. Each test below therefore proves the harness reacts to one specific
# regression the new pipeline could introduce.
#
# One invariant is deliberately NOT re-asserted here: "the stacker never calls torch.cat".
# Banning it across this path would also ban `deepseek_v4_hadamard_idx`, which builds its
# matrix with `torch.cat` and is called for every CSA layer, so the ban only makes sense
# where the packer is mocked out — which is where test_model_components.py already has it.
from __future__ import annotations

from pypto_serving.model.deepseek import weight_loader


def test_stacked_pack_is_reproducible(deepseek_checkpoint, fingerprint_tensors):
    """Same checkpoint twice -> identical bytes. Without this the rest means nothing."""
    checkpoint = deepseek_checkpoint()

    first = fingerprint_tensors(checkpoint.load_stacked().tensors)
    second = fingerprint_tensors(checkpoint.load_stacked().tensors)

    assert first == second
    assert first, "the fingerprint must not be empty"


def test_fingerprint_detects_a_reordered_layer_stack(deepseek_checkpoint, fingerprint_tensors):
    """Swapping two layers' contents must change the fingerprint.

    Layer order is the stacked layout's whole meaning — the fused kernels index a layer by
    its offset in the slab — so a stacker that concatenated in the wrong order would still
    produce the right shapes and dtypes. This is the test that makes such a bug visible.
    """
    baseline = deepseek_checkpoint()
    swapped = deepseek_checkpoint(layer_seeds={0: 1, 1: 0})

    assert fingerprint_tensors(baseline.load_stacked().tensors) != fingerprint_tensors(
        swapped.load_stacked().tensors
    )


def test_group_slabs_only_follow_their_own_attention_kind(deepseek_checkpoint, fingerprint_tensors):
    """CSA slabs must depend on compress_ratio==4 layers only, HCA on ==128 only.

    Group placement is hand-coded today (first-appearance order over three groups). If the
    generic stacker mixed the groups up, shapes would still match; only the provenance of
    the bytes would change, which is what this asserts.
    """
    ratios = (0, 4, 128, 4)
    baseline = deepseek_checkpoint(compress_ratios=ratios)
    # Perturb one CSA layer (index 1, ratio 4) and one HCA layer (index 2, ratio 128).
    csa_perturbed = deepseek_checkpoint(compress_ratios=ratios, layer_seeds={1: 40})
    hca_perturbed = deepseek_checkpoint(compress_ratios=ratios, layer_seeds={2: 41})

    base = fingerprint_tensors(baseline.load_stacked().tensors)
    csa = fingerprint_tensors(csa_perturbed.load_stacked().tensors)
    hca = fingerprint_tensors(hca_perturbed.load_stacked().tensors)

    csa_names = set(weight_loader.DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES) & set(base)
    hca_names = set(weight_loader.DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES) & set(base)
    assert csa_names and hca_names, "the fixture must exercise both groups"

    changed_by_csa = {name for name in base if base[name] != csa[name]}
    changed_by_hca = {name for name in base if base[name] != hca[name]}

    assert csa_names & changed_by_csa, "perturbing a CSA layer must move CSA slabs"
    assert not (hca_names & changed_by_csa), "a CSA layer must not reach the HCA slabs"
    assert hca_names & changed_by_hca, "perturbing an HCA layer must move HCA slabs"
    assert not (csa_names & changed_by_hca), "an HCA layer must not reach the CSA slabs"


def test_stacked_slabs_keep_the_rank_axis_and_stay_contiguous(deepseek_checkpoint):
    """Every slab is ``[ranks, ...]`` and contiguous — both are upload preconditions.

    ``alloc_stacked_tensor`` shards on the leading axis and requires contiguity, so these
    are not cosmetic: losing either breaks the resident-weight upload rather than the math.
    """
    checkpoint = deepseek_checkpoint()

    stacked = checkpoint.load_stacked()

    assert stacked.tensors
    for name, tensor in stacked.tensors.items():
        assert tensor.shape[0] == checkpoint.ranks, f"{name} lost its rank axis: {tuple(tensor.shape)}"
        assert tensor.is_contiguous(), f"{name} is not contiguous"
