"""Tests for the sequence-model harness.

The two things worth guarding here are protocol correctness, not model quality:

- **The engine split must not leak.** If one engine's cycles land in both train and
  validation, val RMSE becomes meaningless and every model looks good.
- **Windows must stay inside one engine**, and ``last_only`` must reproduce exactly the
  last-cycle scoring protocol the RandomForest baseline uses, or the two model families
  are not comparable.

Skipped when torch is unavailable so the core suite still runs on a numpy-only install.
"""

import json
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="sequence models need torch")

from src.features import build_xy  # noqa: E402
from src.seq_models import ARCHITECTURES, make_seq_model  # noqa: E402
from src.train_seq import WindowDataset, pick_device, set_seed, split_engines  # noqa: E402

SEQ_LEN = 8


@pytest.fixture
def featured(small_frames):
    train, test = small_frames
    train_f, test_f, feat_cols, _ = build_xy(train, test)
    return train_f, test_f, feat_cols


class TestSplitEngines:
    def test_no_engine_in_both_splits(self, featured):
        train_f, _, _ = featured
        tr, val, val_units = split_engines(train_f, val_frac=0.5, seed=0)
        assert set(tr["unit"]) & set(val["unit"]) == set()

    def test_every_row_accounted_for(self, featured):
        train_f, _, _ = featured
        tr, val, _ = split_engines(train_f, val_frac=0.5, seed=0)
        assert len(tr) + len(val) == len(train_f)

    def test_val_units_reported_match_val_frame(self, featured):
        train_f, _, _ = featured
        _, val, val_units = split_engines(train_f, val_frac=0.5, seed=0)
        assert sorted(val["unit"].unique().tolist()) == val_units

    def test_deterministic_for_a_seed(self, featured):
        train_f, _, _ = featured
        a = split_engines(train_f, val_frac=0.5, seed=7)[2]
        b = split_engines(train_f, val_frac=0.5, seed=7)[2]
        assert a == b

    def test_always_at_least_one_val_engine(self, featured):
        """A tiny val_frac must not silently produce an empty validation set."""
        train_f, _, _ = featured
        _, val, val_units = split_engines(train_f, val_frac=0.001, seed=0)
        assert len(val_units) >= 1
        assert len(val) > 0

    def test_whole_engines_only(self, featured):
        """Each engine's full cycle history must move together."""
        train_f, _, _ = featured
        tr, val, _ = split_engines(train_f, val_frac=0.5, seed=0)
        for unit in val["unit"].unique():
            expected = (train_f["unit"] == unit).sum()
            assert (val["unit"] == unit).sum() == expected


class TestWindowDataset:
    def test_window_shape(self, featured):
        train_f, _, feat_cols = featured
        ds = WindowDataset(train_f, feat_cols, seq_len=SEQ_LEN)
        x, y = ds[0]
        assert x.shape == (SEQ_LEN, len(feat_cols))
        assert y.ndim == 0

    def test_one_window_per_cycle(self, featured):
        train_f, _, feat_cols = featured
        ds = WindowDataset(train_f, feat_cols, seq_len=SEQ_LEN)
        assert len(ds) == len(train_f)

    def test_early_windows_are_left_padded_with_zeros(self, featured):
        train_f, _, feat_cols = featured
        ds = WindowDataset(train_f, feat_cols, seq_len=SEQ_LEN)
        x, _ = ds[0]  # first cycle of the first engine -> 7 padding rows
        assert torch.allclose(x[: SEQ_LEN - 1], torch.zeros(SEQ_LEN - 1, len(feat_cols)))
        assert not torch.allclose(x[-1], torch.zeros(len(feat_cols)))

    def test_full_windows_have_no_padding(self, featured):
        train_f, _, feat_cols = featured
        ds = WindowDataset(train_f, feat_cols, seq_len=SEQ_LEN)
        x, _ = ds[SEQ_LEN]  # deep enough into engine 1 to be a complete window
        assert not (x == 0).all(dim=1).any()

    def test_windows_do_not_cross_engine_boundary(self, featured):
        """The first window of engine 2 must be padded, not filled with engine 1."""
        train_f, _, feat_cols = featured
        ds = WindowDataset(train_f, feat_cols, seq_len=SEQ_LEN)
        n_engine1 = (train_f["unit"] == 1).sum()
        x, _ = ds[n_engine1]  # first row of engine 2
        assert torch.allclose(x[: SEQ_LEN - 1], torch.zeros(SEQ_LEN - 1, len(feat_cols)))

    def test_targets_match_source_rul(self, featured):
        train_f, _, feat_cols = featured
        ds = WindowDataset(train_f, feat_cols, seq_len=SEQ_LEN)
        ordered = train_f.sort_values(["unit", "cycle"])["RUL"].to_numpy()
        got = np.array([ds[i][1].item() for i in range(len(ds))])
        np.testing.assert_allclose(got, ordered, rtol=1e-6)

    def test_last_only_gives_one_window_per_engine(self, featured):
        _, test_f, feat_cols = featured
        ds = WindowDataset(test_f, feat_cols, seq_len=SEQ_LEN, last_only=True)
        assert len(ds) == test_f["unit"].nunique()
        assert ds.units == sorted(test_f["unit"].unique().tolist())

    def test_last_only_targets_are_the_final_cycle_rul(self, featured):
        """This is the scoring protocol — it must match the baseline's rows exactly."""
        _, test_f, feat_cols = featured
        ds = WindowDataset(test_f, feat_cols, seq_len=SEQ_LEN, last_only=True)
        idx = test_f.groupby("unit")["cycle"].idxmax()
        expected = test_f.loc[idx].sort_values("unit")["RUL"].to_numpy()
        got = np.array([ds[i][1].item() for i in range(len(ds))])
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_last_only_window_ends_on_the_final_observation(self, featured):
        _, test_f, feat_cols = featured
        ds = WindowDataset(test_f, feat_cols, seq_len=SEQ_LEN, last_only=True)
        x, _ = ds[0]
        unit = ds.units[0]
        g = test_f[test_f["unit"] == unit].sort_values("cycle")
        np.testing.assert_allclose(x[-1].numpy(), g[feat_cols].to_numpy()[-1], rtol=1e-5)


class TestArchitectures:
    @pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
    def test_output_shape_is_one_prediction_per_window(self, arch):
        model = make_seq_model(arch, n_features=6, hidden=8, layers=2)
        out = model(torch.randn(4, SEQ_LEN, 6))
        assert out.shape == (4,)

    @pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
    def test_handles_varying_sequence_length(self, arch):
        """The CNN pools over time, so no architecture may hard-code seq_len."""
        model = make_seq_model(arch, n_features=6, hidden=8, layers=2)
        model.eval()
        for seq_len in (5, 20, 47):
            assert model(torch.randn(2, seq_len, 6)).shape == (2,)

    @pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
    def test_prediction_depends_on_the_whole_window(self, arch):
        """A model that ignores most of its window would still pass a shape check.

        Perturbing a mid-window timestep must move the prediction. This catches an
        aggregation that silently reads only one position (e.g. indexing the first
        conv output instead of pooling over time), which is shape-identical and
        therefore invisible to the tests above.
        """
        set_seed(0)
        model = make_seq_model(arch, n_features=6, hidden=8, layers=2)
        model.eval()
        x = torch.randn(2, SEQ_LEN, 6)
        with torch.no_grad():
            before = model(x)
            bumped = x.clone()
            bumped[:, SEQ_LEN // 2, :] += 5.0
            after = model(bumped)
        assert not torch.allclose(
            before, after, atol=1e-6
        ), f"{arch} prediction ignores the middle of the window"

    @pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
    def test_gradients_flow(self, arch):
        model = make_seq_model(arch, n_features=6, hidden=8, layers=2)
        model(torch.randn(4, SEQ_LEN, 6)).sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_single_layer_does_not_request_rnn_dropout(self):
        """torch warns and ignores dropout on a 1-layer RNN; we must not ask for it."""
        for arch in ("lstm", "gru"):
            model = make_seq_model(arch, n_features=4, hidden=8, layers=1, dropout=0.5)
            rnn = getattr(model, "lstm", None) or model.gru
            assert rnn.dropout == 0.0

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="unknown arch"):
            make_seq_model("transformer", n_features=4)


class TestReproducibility:
    def test_set_seed_makes_init_deterministic(self):
        set_seed(123)
        a = make_seq_model("lstm", 6, hidden=8).state_dict()
        set_seed(123)
        b = make_seq_model("lstm", 6, hidden=8).state_dict()
        for k in a:
            assert torch.allclose(a[k], b[k])

    def test_pick_device_returns_a_device(self):
        assert isinstance(pick_device(), torch.device)
        assert pick_device("cpu").type == "cpu"


class TestAttentionInterpretability:
    """`src/interpret.py` treats the attention weights as the model's readout.

    These pin the properties that claim depends on. If any of them breaks, the
    interpretability report becomes confidently wrong rather than failing.
    """

    def _model(self):
        set_seed(0)
        return make_seq_model("attention", n_features=6, hidden=8, layers=2)

    def test_weights_are_one_per_cycle(self):
        w = self._model().attention_weights(torch.randn(4, SEQ_LEN, 6))
        assert w.shape == (4, SEQ_LEN)

    def test_weights_form_a_distribution(self):
        """Sum to 1 and non-negative — otherwise "share of attention" is meaningless."""
        w = self._model().attention_weights(torch.randn(5, SEQ_LEN, 6))
        assert torch.allclose(w.sum(dim=1), torch.ones(5), atol=1e-5)
        assert (w >= 0).all()

    def test_weights_depend_on_the_input(self):
        """Constant weights would make the attention profile pure decoration."""
        model = self._model()
        a = model.attention_weights(torch.randn(3, SEQ_LEN, 6))
        b = model.attention_weights(torch.randn(3, SEQ_LEN, 6) * 5 + 2)
        assert not torch.allclose(a, b, atol=1e-6)

    def test_prediction_equals_the_attention_weighted_readout(self):
        """The central claim: the forward pass IS the weighted sum, not a proxy for it."""
        model = self._model()
        model.eval()
        x = torch.randn(4, SEQ_LEN, 6)
        with torch.no_grad():
            h, w = model._encode(x)
            manual = model.head(torch.einsum("bth,bt->bh", h, w)).squeeze(-1)
            assert torch.allclose(model(x), manual, atol=1e-6)

    def test_weights_do_not_track_gradients(self):
        """attention_weights is inference-only; it must not retain a graph."""
        w = self._model().attention_weights(torch.randn(2, SEQ_LEN, 6))
        assert not w.requires_grad

    def test_single_layer_encoder_skips_rnn_dropout(self):
        model = make_seq_model("attention", n_features=4, hidden=8, layers=1, dropout=0.5)
        assert model.encoder.dropout == 0.0


class TestArchRegistryIsSingleSourceOfTruth:
    """Registering an architecture must reach every CLI.

    `attention` was added to ARCHITECTURES and worked correctly as a model, but four
    entry points hardcoded `choices=["lstm", "gru", "cnn"]`, so `--arch attention` was
    rejected at the argparse layer after a 3-minute training run had already succeeded
    elsewhere. These guard against that duplication returning.
    """

    def test_arch_names_matches_the_registry(self):
        from src.seq_models import ARCH_NAMES, ARCHITECTURES

        assert sorted(ARCHITECTURES) == ARCH_NAMES

    def test_no_module_hardcodes_the_arch_list(self):
        import pathlib
        import re

        offenders = []
        for path in sorted(pathlib.Path("src").glob("*.py")):
            text = path.read_text()
            # a literal list/tuple of arch names used as choices, rather than ARCH_NAMES
            if re.search(r'choices\s*=\s*[\[(]\s*["\']lstm["\']', text):
                offenders.append(path.name)
        assert not offenders, f"hardcoded arch choices in {offenders}; use ARCH_NAMES"

    @pytest.mark.parametrize("module", ["train_seq", "sweep", "ensemble", "variance"])
    def test_every_cli_accepts_every_registered_arch(self, module):
        """Parse-level check: each CLI's choices must equal the registry."""
        import importlib

        from src.seq_models import ARCH_NAMES

        mod = importlib.import_module(f"src.{module}")
        src = pathlib.Path(mod.__file__).read_text()
        assert "ARCH_NAMES" in src, f"src/{module}.py does not use ARCH_NAMES"
        for name in ARCH_NAMES:
            # nothing should be excluding an arch by name
            assert f'choices=["{name}"' not in src


class TestVarianceResultsMerge:
    """`variance.json` accumulates across runs instead of being overwritten.

    Before this, `--archs lstm gru` then `--archs attention cnn` left only the second pair,
    silently discarding the first run's compute and stranding the docs section citing it.
    """

    def test_second_run_preserves_the_first(self, tmp_path):
        from src.variance import merge_results

        path = tmp_path / "variance.json"
        merge_results(path, {"lstm": {"x": 1}, "gru": {"x": 2}})
        merged = merge_results(path, {"attention": {"x": 3}, "cnn": {"x": 4}})

        assert sorted(merged) == ["attention", "cnn", "gru", "lstm"]
        assert merged["lstm"] == {"x": 1}, "first run's results must survive"

    def test_rerunning_an_arch_replaces_only_that_arch(self, tmp_path):
        from src.variance import merge_results

        path = tmp_path / "variance.json"
        merge_results(path, {"lstm": {"x": 1}, "gru": {"x": 2}})
        merged = merge_results(path, {"gru": {"x": 99}})

        assert merged["gru"] == {"x": 99}, "re-run arch should be replaced"
        assert merged["lstm"] == {"x": 1}, "untouched arch should be preserved"

    def test_written_file_matches_the_return_value(self, tmp_path):
        from src.variance import merge_results

        path = tmp_path / "variance.json"
        merge_results(path, {"lstm": {"x": 1}})
        merged = merge_results(path, {"cnn": {"x": 2}})
        assert json.loads(path.read_text()) == merged

    def test_corrupt_file_does_not_lose_this_run(self, tmp_path, capsys):
        """Losing the results being written now would be worse than losing the old ones."""
        from src.variance import merge_results

        path = tmp_path / "variance.json"
        path.write_text("{not json")
        merged = merge_results(path, {"gru": {"x": 1}})

        assert merged == {"gru": {"x": 1}}
        assert "unreadable" in capsys.readouterr().out

    def test_non_object_file_is_replaced(self, tmp_path, capsys):
        from src.variance import merge_results

        path = tmp_path / "variance.json"
        path.write_text("[1, 2, 3]")
        merged = merge_results(path, {"cnn": {"x": 1}})

        assert merged == {"cnn": {"x": 1}}
        assert "not an object" in capsys.readouterr().out
