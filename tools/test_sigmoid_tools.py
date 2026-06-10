import json

import matplotlib
import numpy as np

matplotlib.use("Agg")

from tools import plot_all_sigmoids, refit_sigmoids


def _cell(**overrides):
    cell = {
        "dataset": "sklearn",
        "method": "wanda",
        "H": 32,
        "L": 3,
        "n_params": 1000,
        "A_inf": 0.95,
        "s_0": 0.08,
        "beta": 30.0,
        "R2": 0.99,
        "densities": np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.00]),
        "accs_mean": np.array([0.11, 0.20, 0.61, 0.91, 0.95, 0.95]),
    }
    cell.update(overrides)
    return cell


def test_load_cells_discovers_current_assets_layout(tmp_path):
    results_dir = (
        tmp_path
        / "assets"
        / "unstructured_pruning"
        / "sklearn_wanda"
    )
    results_dir.mkdir(parents=True)
    rows = []
    for repeat in (0, 1):
        rows.append(
            {
                "H": 32,
                "L": 3,
                "repeat": repeat,
                "n_params": 1000,
                "densities": [0.01, 0.05, 0.10, 0.20],
                "accs_mean": [0.11, 0.20, 0.61, 0.91],
                "sigmoid_A_inf_v2": 0.95,
                "sigmoid_s_0_v2": 0.08,
                "sigmoid_beta_v2": 30.0,
                "sigmoid_R2_v2": 0.99,
            }
        )
    (results_dir / "scaling_results.json").write_text(json.dumps(rows))

    cells = plot_all_sigmoids._load_cells(tmp_path)

    assert len(cells) == 1
    assert cells[0]["dataset"] == "sklearn"
    assert cells[0]["method"] == "wanda"
    np.testing.assert_allclose(cells[0]["accs_mean"], rows[0]["accs_mean"])


def test_fit_curve_never_evaluates_negative_density():
    cell = _cell(s_0=0.03)

    x, accuracy = plot_all_sigmoids._physical_fit_curve(
        cell, x_min=-0.5, x_max=0.5, n_x=101
    )

    assert len(x) == len(accuracy)
    assert np.all(cell["s_0"] + x >= 0.0)
    assert x[0] == -cell["s_0"]
    assert x[-1] == 0.5


def test_canonical_refit_uses_fixed_floor_and_physical_s0():
    densities = np.array([0.01, 0.03, 0.06, 0.10, 0.20, 0.40, 0.80, 1.00])
    accuracies = refit_sigmoids.sigmoid(
        densities, A_inf=0.94, s_0=0.12, beta=35.0
    )

    result = refit_sigmoids.fit_sigmoid(densities, accuracies)

    assert result is not None
    assert result["sigmoid_s_0_v2"] >= 0.0
    assert result["sigmoid_A_inf_v2"] > refit_sigmoids.A_FLOOR
    assert "sigmoid_A_0" not in result


def test_combined_figure_puts_raw_data_on_main_axes_and_fits_in_inset():
    cells = [
        _cell(),
        _cell(
            dataset="mnist28",
            method="magnitude",
            n_params=2000,
            s_0=0.12,
            beta=45.0,
        ),
    ]

    fig, main_ax, inset_ax = plot_all_sigmoids.build_combined_figure(
        cells, x_min=-0.5, x_max=0.5, n_x=101
    )
    try:
        main_gids = [line.get_gid() for line in main_ax.lines]
        inset_gids = [line.get_gid() for line in inset_ax.lines]
        assert main_gids.count("raw-data") == len(cells)
        assert "sigmoid-fit" not in main_gids
        assert inset_gids.count("sigmoid-fit") == len(cells)
        assert "raw-data" not in inset_gids
        assert inset_ax.get_xlim() == (-0.5, 0.5)
        assert inset_ax.get_title() == "Fitted accuracy curves"
        for cell, line in zip(cells, inset_ax.lines[: len(cells)]):
            x = line.get_xdata()
            assert x[0] == -cell["s_0"]
            assert x[-1] == 0.5
    finally:
        matplotlib.pyplot.close(fig)
