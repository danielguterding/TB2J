"""TBGreen keeps H(k) of all k-points only on request.

``_prepare_eigen`` accumulates the k average H0 of the Hamiltonian while it
diagonalizes; the array of H(k) for all k-points (as large as the eigenvectors)
is only stored with ``saveH=True``. Eigenvalues, eigenvectors, S(k) and H0 must
not depend on that choice.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sl

from TB2J.green import TBGreen

_NB = 6
_RLIST = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]


class _Model:
    """Small tight-binding model with k-dependent H(k) (and S(k) if non-orthogonal)."""

    R2kfactor = 2j * np.pi
    norb = _NB
    nbasis = _NB
    nel = 4

    def __init__(self, orthogonal, seed=0):
        rng = np.random.default_rng(seed)
        self.is_orthogonal = orthogonal
        a = rng.standard_normal((len(_RLIST), _NB, _NB))
        self.HR = {R: a[i] + 0.3 * i * a[i].T for i, R in enumerate(_RLIST)}
        self.SR = {
            R: (np.eye(_NB) if R == (0, 0, 0) else 0.05 * a[i])
            for i, R in enumerate(_RLIST)
        }

    def _k_sum(self, MR, k):
        M = sum(np.exp(2j * np.pi * np.dot(R, k)) * m for R, m in MR.items())
        return 0.5 * (M + M.conj().T)

    def HSE_k(self, kpt, convention=2):
        Hk = self._k_sum(self.HR, kpt)
        if self.is_orthogonal:
            evals, evecs = np.linalg.eigh(Hk)
            return Hk, None, evals, evecs
        Sk = self._k_sum(self.SR, kpt)
        evals, evecs = sl.eigh(Hk, Sk)
        return Hk, Sk, evals, evecs


@pytest.mark.parametrize("orthogonal", [True, False])
def test_H0_without_storing_Hk(orthogonal):
    model = _Model(orthogonal)
    # efermi well above the band so that the default energy window keeps all states
    G = TBGreen(tbmodel=model, kmesh=[2, 2, 1], efermi=10.0)
    assert G.H is None

    ref = [model.HSE_k(k) for k in G.kpts]
    np.testing.assert_allclose(G.H0, np.mean([r[0] for r in ref], axis=0), atol=1e-13)
    np.testing.assert_allclose(G.evals, [r[2] for r in ref], atol=1e-12)
    np.testing.assert_allclose(G.evecs, [r[3] for r in ref], atol=1e-12)
    if orthogonal:
        assert G.S is None
    else:
        np.testing.assert_allclose(G.S, [r[1] for r in ref], atol=1e-13)

    H0 = G.H0.copy()
    G._prepare_eigen(saveH=True)
    np.testing.assert_allclose(G.H, [r[0] for r in ref], atol=1e-13)
    np.testing.assert_allclose(G.H0, H0, atol=1e-13)
