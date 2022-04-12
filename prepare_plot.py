import plotly.graph_objects as go
import numpy as np
from calc_xrr import calc_reflectivity, prep_model


def get_data(data_dict):
    q = data_dict["experiment"]["q"]
    xrr = data_dict["experiment"]["data"]
    thickness = data_dict["fit"]["Film_thickness"]

    return q, xrr, thickness


def prepare_figure(dataset, q_fit, label_prefix="film thickness"):

    # prepare data
    x, data, labels = get_data(dataset)
    fits = calc_reflectivity(*prep_model(q_fit, dataset))

    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for step, label in enumerate(labels):
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(width=2),
                marker={"color": "#00CED1", "size": 10},
                mode="markers",
                name="exp.",
                x=x.astype(float),
                y=data[step].astype(float),
            )
        )
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(width=2),
                marker={"color": "#00EE00", "size": 10},
                mode="lines",
                name="fit",
                x=q_fit.astype(float),
                y=fits[step].astype(float),
            )
        )
    fig.update_yaxes(
        title_text="X-ray reflectivity",
        type="log",
        range=[np.floor(np.log10(np.min(data))), 0],
    )
    fig.update_xaxes(title_text="momentum transfer q [1/Ang]")

    # Make first trace visible
    fig.data[0].visible = True
    fig.data[1].visible = True

    # Create and add slider
    steps = []
    for i in range(len(data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": label_prefix + f": {labels[i]:.2f} Ang."},
            ],  # layout attribute
            label=f"scan {i}",
        )
        step["args"][0]["visible"][2 * i] = True  # Toggle trace to "visible"
        step["args"][0]["visible"][2 * i + 1] = True  # Toggle trace to "visible"
        steps.append(step)

    sliders = [dict(active=0, steps=steps)]

    fig.update_layout(sliders=sliders)

    return fig
