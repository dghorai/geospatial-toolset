# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 19:26:11 2024

@author: Debabrata Ghorai, Ph.D.

Define progressbar functions.

"""

import sys

from streamlit.runtime.scriptrunner import get_script_run_ctx


def is_streamlit():
    """Returns True if the code is running inside a Streamlit app."""
    try:
        return get_script_run_ctx() is not None
    except ImportError:
        return False


# Import the correct progress bar
def pbar_object():
    print(is_streamlit())

    pbar = None
    if not is_streamlit():
        from tqdm.notebook import tqdm  # Specialized for Jupyter/VS Code
        pbar = tqdm(total=100, desc="Executing Tasks")
        
    return pbar


# Update logic that works for both
def update_bar(increment, state, pbar, status):
    # Add the incoming increment to the total
    state["current_total"] += float(increment)
    # Clamp between 0.0 and 1.0 for Streamlit
    percent = max(0.0, min(state["current_total"] / 100, 1.0))
    # Update the UI
    pbar.progress(percent)
    status.text(f"Progress: {int(percent * 100)}%")
    return
