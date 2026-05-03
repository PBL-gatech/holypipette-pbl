import numpy as np
import h5py
import pytest

from experiments.SimpleDatasetBuilder import PretrainTargetConfig, SimpleDatasetBuilder
from patcherbot.utils.resistance_smoothing import (
    CAUSAL_LOG_MEDIAN_EMA_METHOD,
    DEFAULT_EMA_ALPHA,
    DEFAULT_MEDIAN_WINDOW,
    DEFAULT_RESISTANCE_FLOOR,
    CausalLogMedianEmaSmoother,
    causal_log_median_ema,
)


def _target(
    *,
    horizon=1,
    threshold=0.02,
    smoothing_window=1,
    resistance_floor=1e-3,
    ema_alpha=1.0,
    mask_command_interventions=False,
):
    return PretrainTargetConfig(
        name=f"trend_{horizon}",
        horizon=horizon,
        threshold=threshold,
        label_key=f"pretrain/trend_{horizon}",
        mask_key=f"pretrain/mask_{horizon}",
        smoothing_window=smoothing_window,
        resistance_floor=resistance_floor,
        ema_alpha=ema_alpha,
        mask_command_interventions=mask_command_interventions,
    )


def _builder(target, dataset_path=None):
    builder = object.__new__(SimpleDatasetBuilder)
    builder.pretrain_targets = [target]
    if dataset_path is not None:
        builder.dataset_path = dataset_path
    return builder


def test_pretrain_target_defaults_use_causal_smoother():
    target = PretrainTargetConfig(name="trend_10", horizon=10)

    assert target.smoothing_method == CAUSAL_LOG_MEDIAN_EMA_METHOD
    assert target.smoothing_window == DEFAULT_MEDIAN_WINDOW
    assert target.resistance_floor == DEFAULT_RESISTANCE_FLOOR
    assert target.ema_alpha == DEFAULT_EMA_ALPHA


def test_causal_log_median_ema_initializes_and_updates_ema():
    resistance = np.asarray([100.0, 200.0, 400.0])

    smoothed = causal_log_median_ema(
        resistance,
        window=1,
        alpha=0.25,
        resistance_floor=1e-3,
    )

    log_values = np.log(resistance)
    expected = np.asarray(
        [
            log_values[0],
            0.25 * log_values[1] + 0.75 * log_values[0],
            0.25 * log_values[2]
            + 0.75 * (0.25 * log_values[1] + 0.75 * log_values[0]),
        ]
    )
    np.testing.assert_allclose(smoothed, expected)


def test_incremental_smoother_matches_batch_smoother():
    resistance = np.asarray([100.0, 120.0, np.nan, 121.0, 5000.0, 122.0])
    smoother = CausalLogMedianEmaSmoother(
        window=3,
        alpha=0.3,
        resistance_floor=1e-3,
    )

    online = np.asarray([smoother.update(value) for value in resistance])
    batch = causal_log_median_ema(
        resistance,
        window=3,
        alpha=0.3,
        resistance_floor=1e-3,
    )

    np.testing.assert_allclose(online, batch, equal_nan=True)


def test_causal_smoother_does_not_use_future_spikes():
    prefix = np.asarray([100.0, 101.0, 102.0, 103.0])
    with_future_spike = np.concatenate([prefix, np.asarray([1_000_000.0])])
    without_future_spike = np.concatenate([prefix, np.asarray([104.0])])

    spike_trace = causal_log_median_ema(
        with_future_spike,
        window=5,
        alpha=0.3,
        resistance_floor=1e-3,
    )
    baseline_trace = causal_log_median_ema(
        without_future_spike,
        window=5,
        alpha=0.3,
        resistance_floor=1e-3,
    )

    np.testing.assert_allclose(spike_trace[: prefix.size], baseline_trace[: prefix.size])


def test_future_log_trend_labels_classes_and_tail_mask():
    target = _target(horizon=1, threshold=0.02)
    builder = _builder(target)
    resistance = np.asarray([100.0, 103.0, 103.0, 100.0, 100.0])

    datasets, _ = builder.build_pretrain_datasets(resistance)

    np.testing.assert_array_equal(
        datasets[target.label_key].reshape(-1),
        np.asarray([2, 1, 0, 1, 1]),
    )
    np.testing.assert_array_equal(
        datasets[target.mask_key].reshape(-1),
        np.asarray([1, 1, 1, 1, 0], dtype=np.float32),
    )


def test_future_log_trend_floors_finite_zero_resistance():
    target = _target(horizon=1, threshold=0.02)
    builder = _builder(target)
    resistance = np.asarray([0.0, 0.002, np.nan, 0.004])

    datasets, attrs = builder.build_pretrain_datasets(resistance)

    np.testing.assert_array_equal(
        datasets[target.label_key].reshape(-1),
        np.asarray([2, 1, 1, 1]),
    )
    np.testing.assert_array_equal(
        datasets[target.mask_key].reshape(-1),
        np.asarray([1, 0, 0, 0], dtype=np.float32),
    )
    assert attrs[target.label_key]["smoothing_method"] == CAUSAL_LOG_MEDIAN_EMA_METHOD
    assert attrs[target.label_key]["resistance_floor"] == 1e-3
    assert attrs[target.label_key]["ema_alpha"] == 1.0


def test_future_log_trend_masks_nonfinite_endpoints():
    target = _target(horizon=1, threshold=0.02)
    builder = _builder(target)
    resistance = np.asarray([100.0, np.nan, 103.0, np.inf, 106.0])

    datasets, _ = builder.build_pretrain_datasets(resistance)

    np.testing.assert_array_equal(
        datasets[target.mask_key].reshape(-1),
        np.asarray([0, 0, 0, 0, 0], dtype=np.float32),
    )


def test_future_log_trend_masks_command_interventions():
    target = _target(horizon=2, threshold=0.02, mask_command_interventions=True)
    builder = _builder(target)
    resistance = np.asarray([100.0, 103.0, 106.0, 109.0, 112.0])
    commanded_pressure = np.asarray([0.0, 0.0, 5.0, 5.0, 5.0])
    atm_state = np.zeros(5)

    datasets, _ = builder.build_pretrain_datasets(
        resistance,
        commanded_pressure,
        atm_state,
    )

    np.testing.assert_array_equal(
        datasets[target.mask_key].reshape(-1),
        np.asarray([0, 0, 1, 0, 0], dtype=np.float32),
    )


def test_pretrain_writer_overwrites_only_owned_paths(tmp_path):
    target = _target()
    hdf5_path = tmp_path / "demo.hdf5"
    labels = np.asarray([[2], [2], [2], [2], [1]], dtype=np.int64)
    masks = np.asarray([[1], [1], [1], [1], [0]], dtype=np.float32)

    with h5py.File(hdf5_path, "w") as hf:
        demo = hf.create_group("data/demo_0")
        demo.create_group("pretrain").create_dataset("unrelated", data=np.asarray([[9]]))
        demo.create_dataset(target.label_key, data=np.zeros((5, 1), dtype=np.int64))
        SimpleDatasetBuilder._write_pretrain_datasets(
            demo,
            {target.label_key: labels, target.mask_key: masks},
            {target.label_key: {"role": "label"}, target.mask_key: {"role": "mask"}},
        )

    with h5py.File(hdf5_path, "r") as hf:
        demo = hf["data/demo_0"]
        np.testing.assert_array_equal(demo[target.label_key][()], labels)
        np.testing.assert_array_equal(demo[target.mask_key][()], masks)
        np.testing.assert_array_equal(demo["pretrain/unrelated"][()], np.asarray([[9]]))
        assert demo[target.label_key].attrs["role"] == "label"


def test_debug_single_trajectory_rewrite_keeps_one_compact_demo(tmp_path):
    hdf5_path = tmp_path / "debug.hdf5"
    original_actions = {
        f"demo_{idx}": np.full((3, 2), idx, dtype=np.float32)
        for idx in range(4)
    }

    with h5py.File(hdf5_path, "w") as hf:
        data = hf.create_group("data")
        data.attrs["num_demos"] = len(original_actions)
        for demo_name, actions in original_actions.items():
            demo = data.create_group(demo_name)
            demo.attrs["num_samples"] = actions.shape[0]
            demo.attrs["split"] = "train"
            demo.create_dataset("actions", data=actions)
        mask = hf.create_group("mask")
        mask.create_dataset("train", data=np.asarray(["demo_0", "demo_1"], dtype="S"))
        mask.create_dataset("valid", data=np.asarray(["demo_2", "demo_3"], dtype="S"))

    builder = object.__new__(SimpleDatasetBuilder)
    builder.dataset_name = hdf5_path.name
    builder.dataset_path = hdf5_path
    builder._split_keys = {"train": ["demo_0", "demo_1"], "valid": ["demo_2", "demo_3"]}
    builder._state_contexts = {}
    builder.pretrain_targets = []
    builder._write_metadata_files = lambda: None

    selected = builder.write_debug_single_trajectory_dataset(random_seed=5)
    selected_demo = selected[hdf5_path.name]

    with h5py.File(hdf5_path, "r") as hf:
        assert sorted(hf["data"].keys()) == ["demo_0"]
        assert int(hf["data"].attrs["num_demos"]) == 1
        assert hf["data/demo_0"].attrs["split"] == "train"
        np.testing.assert_array_equal(
            hf["data/demo_0/actions"][()],
            original_actions[selected_demo],
        )
        assert [value.decode() for value in hf["mask/train"][()]] == ["demo_0"]
        assert [value.decode() for value in hf["mask/valid"][()]] == ["demo_0"]


def _write_valid_pretrain_hdf5(path, target):
    labels = np.asarray([[2], [2], [2], [2], [1]], dtype=np.int64)
    masks = np.asarray([[1], [1], [1], [1], [0]], dtype=np.float32)
    with h5py.File(path, "w") as hf:
        data = hf.create_group("data")
        data.attrs["num_demos"] = 1
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = 5
        demo.create_dataset("actions", data=np.zeros((5, 2), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("resistance", data=np.asarray([[100], [103], [106], [109], [112]], dtype=np.float32))
        SimpleDatasetBuilder._write_pretrain_datasets(
            demo,
            {target.label_key: labels, target.mask_key: masks},
            {
                target.label_key: {"role": "label"},
                target.mask_key: {"role": "mask", "source_endpoint_invalid_count": 0},
            },
        )
        mask = hf.create_group("mask")
        mask.create_dataset("train", data=np.asarray(["demo_0"], dtype="S"))
        mask.create_dataset("valid", data=np.asarray([], dtype="S"))


def test_pretrain_validation_report_accepts_valid_dataset(tmp_path):
    target = _target()
    hdf5_path = tmp_path / "valid.hdf5"
    _write_valid_pretrain_hdf5(hdf5_path, target)
    builder = _builder(target, hdf5_path)

    report = builder.validate_pretrain_targets()

    assert report["demo_count"] == 1
    assert report["targets"][target.name]["valid_labels"] == 4
    assert report["targets"][target.name]["class_histogram"] == {"0": 0, "1": 0, "2": 4}


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda demo, target: demo[target.label_key].__setitem__((0, 0), 5), "outside"),
        (lambda demo, target: demo[target.mask_key].__setitem__((0, 0), 0.5), "outside"),
        (lambda demo, target: demo[target.mask_key].__setitem__((-1, 0), 1), "tail"),
        (lambda demo, target: demo["obs/resistance"].__setitem__((0, 0), np.nan), "source endpoints"),
    ],
)
def test_pretrain_validation_rejects_bad_demo_payloads(tmp_path, mutator, message):
    target = _target()
    hdf5_path = tmp_path / "bad.hdf5"
    _write_valid_pretrain_hdf5(hdf5_path, target)
    with h5py.File(hdf5_path, "a") as hf:
        mutator(hf["data/demo_0"], target)
    builder = _builder(target, hdf5_path)

    with pytest.raises(ValueError, match=message):
        builder.validate_pretrain_targets()


def test_pretrain_validation_rejects_stale_split_key(tmp_path):
    target = _target()
    hdf5_path = tmp_path / "stale.hdf5"
    _write_valid_pretrain_hdf5(hdf5_path, target)
    with h5py.File(hdf5_path, "a") as hf:
        del hf["mask/train"]
        hf["mask"].create_dataset("train", data=np.asarray(["demo_0", "missing_demo"], dtype="S"))
    builder = _builder(target, hdf5_path)

    with pytest.raises(ValueError, match="missing_demo"):
        builder.validate_pretrain_targets()
