
import streamlit as st 
import os
from src.frontend.api_calls import colorizer_model_change_api, colorize_param_change_api
from src.data.config import DEFAULT_CONFIG

def colorizer_model_change(model_config_tab):
    """Handle the colorization model change request."""
    model_paths = os.listdir("models/colorize_models")

    st.selectbox(
        "Select Colorization Model",
        options=model_paths,
        key="colorization_model",
        on_change=colorizer_model_change_api,
        args=(model_config_tab,)
    )


def colorize_param_change(model_config_tab):
    cols = st.columns(2)
    with cols[0]:
        st.number_input(
            "Denoise Sigma",
            min_value=0.0,
            max_value=50.0,
            value=50.0,
            step=5.0,
            key="denoise_sigma",
            on_change=colorize_param_change_api,
            args=(model_config_tab,)
        )
    with cols[1]:
        st.number_input(
            "Image Tile Size",
            min_value=576,
            max_value=1920,
            value=1280,
            step=32,
            key="image_tile_size",
            on_change=colorize_param_change_api,
            args=(model_config_tab,)
        )
#