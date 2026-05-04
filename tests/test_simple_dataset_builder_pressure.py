from contextlib import contextmanager

import numpy as np
import pandas as pd

from experiments.SimpleDatasetBuilder import ActionSelector, ObservationSelector, SimpleDatasetBuilder


def _pressure_builder(action_selector=None):
    builder = object.__new__(SimpleDatasetBuilder)
    builder.action_selector = action_selector or ActionSelector(
        include_stage=False,
        include_pipette=False,
        include_pressure=True,
    )
    return builder


def _pressure_events():
    return pd.DataFrame(
        {
            "timestamp": [0.4, 1.4],
            "event_kind": ["pressure", "atm_state"],
            "commanded_pressure_mbar": [-5.0, np.nan],
            "pressure_atm_state": [np.nan, 1.0],
        }
    )


def test_pressure_command_observation_trace_does_not_use_future_events():
    builder = _pressure_builder()
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )

    commanded_pressure, atm_state = builder.get_attempt_pressure_observation_values(
        graph_values,
        _pressure_events(),
    )

    np.testing.assert_array_equal(commanded_pressure, np.asarray([0.0, -5.0, -5.0]))
    np.testing.assert_array_equal(atm_state, np.asarray([0.0, 0.0, 1.0]))


def test_pressure_action_targets_command_and_atm_after_observation():
    builder = _pressure_builder()
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )

    target_pressure, target_atm_state = builder.get_attempt_pressure_action_values(
        graph_values,
        _pressure_events(),
    )

    np.testing.assert_array_equal(target_pressure, np.asarray([-5.0, -5.0, -5.0]))
    np.testing.assert_array_equal(target_atm_state, np.asarray([0.0, 1.0, 1.0]))


def test_non_gigaseal_pressure_actions_write_raw_commanded_pressure_only():
    builder = _pressure_builder(
        ActionSelector(
            include_stage=False,
            include_pipette=False,
            include_pressure=True,
        )
    )
    builder.use_velocities = False
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )

    actions = builder.get_attempt_actions(
        movement_values,
        attempt_graph_values=graph_values,
        pressure_events=_pressure_events(),
    )

    np.testing.assert_array_equal(
        actions,
        np.asarray(
            [
                [-5.0],
                [-5.0],
                [-5.0],
            ]
        ),
    )
    assert builder._last_action_labels == ["commanded_pressure_mbar"]


def test_gigaseal_force_pressure_actions_prepend_pressure_columns():
    builder = _pressure_builder(
        ActionSelector(
            include_stage=False,
            include_pipette=True,
            include_pressure=False,
        )
    )
    builder.use_velocities = False
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )

    actions = builder.get_attempt_actions(
        movement_values,
        attempt_graph_values=graph_values,
        pressure_events=_pressure_events(),
        force_pressure_action=True,
    )

    np.testing.assert_array_equal(actions[:, :2], np.asarray([[-5.0, 0.0], [-5.0, 1.0], [-5.0, 1.0]]))
    assert builder._last_action_labels[:2] == ["commanded_pressure_mbar", "pressure_atm_state"]


def test_gigaseal_since_last_action_observations_ignore_current_action_target():
    actions = np.asarray(
        [[0.0, 0.0], [-5.0, 0.0], [-5.0, 1.0], [-5.0, 1.0], [-5.0, 1.0]],
        dtype=np.float64,
    )
    labels = ["commanded_pressure_mbar", "pressure_atm_state"]
    timestamps = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    counts = SimpleDatasetBuilder.get_observations_since_last_action_values(
        actions,
        labels,
    )
    elapsed = SimpleDatasetBuilder.get_time_since_last_action_values(
        actions,
        labels,
        timestamps,
    )

    np.testing.assert_array_equal(counts, np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(elapsed, np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]))


def test_gigaseal_pressure_observation_keeps_continuous_graph_pressure():
    builder = object.__new__(SimpleDatasetBuilder)
    builder.dataset_name = "pressure_regression.hdf5"
    builder.load_next_obs = False
    builder.gigaseal_start_trim_enabled = False
    builder.gigaseal_resistance_cutoff_enabled = False
    builder.gigaseal_resistance_cutoff = 1200.0
    builder.val_ratio = 0.0
    builder.omit_stage_movement = False
    builder.skip_invalid_observations = False
    builder.inaction = 0
    builder.inaction_tolerance = 0.0
    builder.rng = np.random.default_rng(0)
    builder._split_keys = {"train": []}
    builder._last_action_labels = []

    builder.observation_selector = ObservationSelector(
        include_pressure=True,
        include_resistance=False,
        include_resistance_slope=False,
        include_current=False,
        include_voltage=False,
        include_stage=False,
        include_pipette=False,
        include_camera=False,
        include_gigaseal_pressure_state=True,
        include_gigaseal_effective_pressure=True,
        include_gigaseal_observations_since_last_action=False,
        include_gigaseal_time_since_last_action=False,
    )
    builder.action_selector = ActionSelector(
        include_stage=False,
        include_pipette=False,
        include_pressure=False,
    )

    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )
    pressure_events = pd.DataFrame(
        {
            "timestamp": [0.4],
            "event_kind": ["pressure"],
            "commanded_pressure_mbar": [50.0],
            "pressure_atm_state": [np.nan],
        }
    )

    builder.load_experiment_data = lambda folder: (graph_values, movement_values)
    builder.get_timestamps_for_all_successful_state_attempts = (
        lambda folder, start, end: {"gigasealing": [(0.0, 2.0)]}
    )
    builder._load_pressure_log_events = lambda folder: pressure_events

    @contextmanager
    def state_context(state_name):
        yield

    builder._use_state_context = state_context
    builder.begin_filter_context = lambda split_label, demo_seed: None
    builder.end_filter_context = lambda: None
    builder._register_processed_folder = lambda folder: False
    builder._write_metadata_files = lambda: None
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )

    def fake_actions(attempt_movement_values, attempt_graph_values=None, **kwargs):
        builder._last_action_labels = []
        builder._last_stage_motion_detected = False
        return np.zeros((attempt_graph_values.shape[0], 0), dtype=np.float64)

    builder.get_attempt_actions = fake_actions

    captured = {}

    def capture_demo(**kwargs):
        captured.update(kwargs)
        return "demo_0"

    builder.add_attempt_demo_to_dataset = capture_demo

    builder.add_demo("fake-folder", record_to_file=True)

    np.testing.assert_array_equal(captured["pressure_values"], graph_values[:, 1])
    np.testing.assert_array_equal(captured["effective_pressure_values"], np.asarray([0.0, 50.0, 50.0]))


def test_gigaseal_pressure_observations_stay_current_while_actions_target_future():
    builder = object.__new__(SimpleDatasetBuilder)
    builder.dataset_name = "pressure_alignment.hdf5"
    builder.load_next_obs = False
    builder.gigaseal_start_trim_enabled = False
    builder.gigaseal_resistance_cutoff_enabled = False
    builder.gigaseal_resistance_cutoff = 1200.0
    builder.val_ratio = 0.0
    builder.omit_stage_movement = False
    builder.skip_invalid_observations = False
    builder.inaction = 0
    builder.inaction_tolerance = 0.0
    builder.use_velocities = False
    builder.rng = np.random.default_rng(0)
    builder._split_keys = {"train": []}
    builder._last_action_labels = []

    builder.observation_selector = ObservationSelector(
        include_pressure=True,
        include_resistance=True,
        include_resistance_slope=False,
        include_current=False,
        include_voltage=False,
        include_stage=False,
        include_pipette=False,
        include_camera=False,
        include_gigaseal_pressure_state=True,
        include_gigaseal_effective_pressure=True,
        include_gigaseal_observations_since_last_action=True,
        include_gigaseal_time_since_last_action=False,
    )
    builder.action_selector = ActionSelector(
        include_stage=False,
        include_pipette=False,
        include_pressure=False,
    )

    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
            [3.0, 1.4, 103.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )
    pressure_events = pd.DataFrame(
        {
            "timestamp": [0.4, 1.4],
            "event_kind": ["pressure", "atm_state"],
            "commanded_pressure_mbar": [50.0, np.nan],
            "pressure_atm_state": [np.nan, 1.0],
        }
    )

    builder.load_experiment_data = lambda folder: (graph_values, movement_values)
    builder.get_timestamps_for_all_successful_state_attempts = (
        lambda folder, start, end: {"gigasealing": [(0.0, 3.0)]}
    )
    builder._load_pressure_log_events = lambda folder: pressure_events

    @contextmanager
    def state_context(state_name):
        yield

    builder._use_state_context = state_context
    builder.begin_filter_context = lambda split_label, demo_seed: None
    builder.end_filter_context = lambda: None
    builder._register_processed_folder = lambda folder: False
    builder._write_metadata_files = lambda: None
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )

    captured = {}

    def capture_demo(**kwargs):
        captured.update(kwargs)
        return "demo_0"

    builder.add_attempt_demo_to_dataset = capture_demo

    builder.add_demo("fake-folder", record_to_file=True)

    np.testing.assert_array_equal(captured["effective_pressure_values"], np.asarray([0.0, 50.0, 0.0, 0.0]))
    np.testing.assert_array_equal(captured["pressure_state_values"], np.asarray([0.0, 0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(
        captured["observations_since_last_action_values"],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        captured["actions"][:, :2],
        np.asarray(
            [
                [50.0, 0.0],
                [50.0, 1.0],
                [50.0, 1.0],
                [50.0, 1.0],
            ]
        ),
    )
    assert builder._last_action_labels[:2] == ["commanded_pressure_mbar", "pressure_atm_state"]
