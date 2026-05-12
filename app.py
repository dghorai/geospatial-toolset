# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 19:26:11 2024

@author: Debabrata Ghorai, Ph.D.

Streamlit Server - An Independent Server for Fronted GUI.

"""

import os
import sys
import streamlit as st

from streamlit.web import cli as stcli
from src.gui.toolbox import my_toolbox
from src.gui.shoreline_transects import gui_shoreline_transects
from consts import PRJ_FLD


# Define your preferred base path
custom_path = os.path.join(PRJ_FLD, "tmp")
os.makedirs(custom_path, exist_ok=True)


# CSS TO REMOVE WHITESPACE, CSS TO ELIMINATE THE GAP
st.markdown("""
    <style>
        /* 1. Global Page Top Padding */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }

        /* 2. Sidebar Top Padding */
        [data-testid="stSidebarNav"] {
            padding-top: 0rem !important;
        }
        [data-testid="stSidebar"] .st-emotion-cache-6qob1r {
            padding-top: 1rem !important;
        }

        /* 3. Eliminate Gap between Navbar and Divider */
        /* Targets the vertical spacing between the nav columns and the line */
        [data-testid="stVerticalBlock"] > div:has(div[data-testid="stHorizontalBlock"]) {
            gap: 0rem !important;
        }

        /* Targets the widget's internal bottom margin */
        [data-testid="stBaseButton-segmented_control"] {
            margin-bottom: 0px !important;
        }
        
        /* 4. Tighten the Divider (The Line) */
        hr {
            margin-top: 0px !important;
            margin-bottom: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)


# MAIN function of this app
def run_app():
    # --- PAGE CONFIG ---
    st.set_page_config(page_title="Geospatial Toolset", layout="wide", page_icon="🗺")
    
    # TOP NAV (Horizontal Right) - We use columns to push the tabs to the right
    col_empty, col_nav = st.columns([3, 2])
    with col_nav:
        # label_visibility is now "visible" by default
        selected_subject = st.segmented_control(
            "Select Subject Area",  # This label will now be visible
            options=list(my_toolbox.keys()),
            selection_mode="single",
            default="🏠 Home"
        )
    # Break page
    st.divider()

    # SIDEBAR (Dynamic based on Navbar selection)
    with st.sidebar:
        st.header(f"{selected_subject}")
        st.subheader("Available Tools")
        # The tools list changes based on what you picked in the Navbar
        selected_tool = st.selectbox(
            "Choose a tool:", 
            options=my_toolbox[selected_subject]
        )
        # status text
        st.info(f"Currently using: {selected_tool}")
    
    # MAIN BODY (Content based on the Tool) - with active_tab:
    st.title(f"Tool: {selected_tool}")
    st.write(f"This is the interface for the **{selected_tool}** tools.")

    # Call Tool-Specific Logic
    if selected_tool == "Extract Shorelines":
        st.button("Run Scrubbing Script")
    elif selected_tool == "Create Transects":
        gui_shoreline_transects(custom_path)         
    elif selected_tool == "Add Transects SeqID":
        st.line_chart([1, 2, 3, 5, 8])
    elif selected_tool == "Shoreline Change Analysis":
        st.button("Shoreline Change Analysis")
    else:
        raise Exception(f"{selected_tool} tools not available in this app.")

    # --- FOOTER ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.caption("Geospatial Toolset | Developed by Debabrata Ghorai, PhD.")

    return


if __name__ == "__main__":
    if st.runtime.exists():
        # If we are already in the Streamlit environment, run the app logic
        run_app()
    else:
        # If we are running 'python app.py', trigger the Streamlit CLI
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

# Note: Press Ctrl+C to stop streamlit server.
# Activate Virtual Env: <virtual env path>\Scripts\activate.ps1
# Run this app:  streamlit run app.py