# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:10:11 2026

@author: Debabrata Ghorai, Ph.D.

GUI for generating shoreline transects.

"""

import os
import streamlit as st
import tempfile

from src.gui.dialog import set_new_shapefile_path
from projects.shoreline.generate_shoreline_transects import create_shoreline_transects
from src.utils.progress_bar import update_bar


def gui_shoreline_transects(custom_path):
    # --- 1. HEADER SECTION ---
    st.title("🌊 Shoreline Transect Generator")
    st.markdown("Generate cross-section transects between onshore and offshore lines.")
    st.divider()
    
    # --- 2. SESSION STATE INITIALIZATION ---
    if 'final_save_path' not in st.session_state:
        st.session_state.final_save_path = None
    
    # --- 3. SAVE LOCATION (Keep it Outside Form) ---
    st.subheader("Step 1: Define Output")
    col_path, col_status = st.columns([1, 2])
    
    with col_path:
        if st.button('📁 Pick Save Location'):
            out_transect_line = set_new_shapefile_path()
            if out_transect_line:
                st.session_state.final_save_path = out_transect_line
    
    with col_status:
        if st.session_state.final_save_path:
            st.info(f"**Saving to:** {os.path.basename(st.session_state.final_save_path)}")
        else:
            st.warning("No save location selected.")

    # create gap
    st.write("") # Spacer
    
    # --- 4. INPUT FORM ---
    st.subheader("Step 2: Upload Data & Configure")
    with st.form("input_form"):
        # Main content columns
        col1, col2 = st.columns(2)
        
        with col1:
            onshore_files = st.file_uploader(
                "Upload Onshore Shapefile (select .shp, .shx, .dbf) OR a GeoJSON",
                type=['shp', 'shx', 'dbf', 'prj', 'geojson'],
                accept_multiple_files=True
            )
            # x_interval = st.number_input("Cross-Section Interval", value=None, placeholder="Type a number...")
            x_interval = st.number_input(
                "Cross-Section Interval", 
                min_value=10, 
                value=1000, 
                help="Distance between transects"
            )
            
        with col2:
            offshore_files = st.file_uploader(
                "Upload Offshore Shapefile (select .shp, .shx, .dbf) OR a GeoJSON",
                type=['shp', 'shx', 'dbf', 'prj', 'geojson'],
                accept_multiple_files=True
            )
            
        # Add a divider to separate inputs from the button
        st.divider()
        
        # Align Run Button to the Right
        # Create two columns for the bottom row
        # Use weights [3, 1] to make the right column smaller
        _, btn_col = st.columns([3, 1])
        with btn_col:
            # MANDATORY: The only button allowed in the form
            submit = st.form_submit_button("🚀 Run Analysis", use_container_width=True)
            
    # --- 5. EXECUTION LOGIC ---
    if submit:
        # Space below the button for the progress area
        st.markdown("<br>", unsafe_allow_html=True)
        
        if onshore_files and offshore_files and st.session_state.final_save_path:
            # The directory will be created INSIDE your custom path
            with tempfile.TemporaryDirectory(dir=custom_path) as tmp_dir:
                
                # --- SAVE ONSHORE ---
                onshore_path = ""
                for f in onshore_files:
                    p = os.path.join(tmp_dir, f.name)
                    with open(p, "wb") as b: b.write(f.getbuffer())
                    if f.name.lower().endswith((".shp", ".geojson")): 
                        onshore_path = p

                # --- SAVE OFFSHORE ---
                offshore_path = ""
                for f in offshore_files:
                    p = os.path.join(tmp_dir, f.name)
                    with open(p, "wb") as b: b.write(f.getbuffer())
                    if f.name.lower().endswith((".shp", ".geojson")): 
                        offshore_path = p
                
                # --- PREPARE OUTPUT ---
                # Use the local path chosen by the user
                final_out = st.session_state.final_save_path.replace('.shp', '_result.shp')
                
                try:
                    # 1. Setup the UI elements
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    # Create a tracker for the total percentage
                    state = {"current_total": 0.0}
                    
                    # CRITICAL: Pass the PATHS (strings), not the file objects
                    create_shoreline_transects(
                        onshore_line=onshore_path, 
                        offshore_line=offshore_path,
                        out_transect_line=final_out,
                        x_interval=x_interval,
                        progress_callback=lambda inc: update_bar(inc, state, progress_bar, status_text)
                    )
                    status_text.empty()
                    st.success(f"Successfully saved to {final_out}")
                except Exception as e:
                    st.error(f"Processing failed: {e}")
        else:
            if not onshore_files: st.error("Missing Onshore files")
            if not offshore_files: st.error("Missing Offshore files")
            if not st.session_state.final_save_path: st.error("Missing Save Location)")
    return
                