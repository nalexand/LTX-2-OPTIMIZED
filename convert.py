import streamlit as st
import pandas as pd
import re
import torch
import os
from safetensors import safe_open
from safetensors.torch import save_file

# Set up the web page
st.set_page_config(page_title="Safetensors Dtype Modifier", layout="wide")
st.title("Safetensors Dtype Modifier")
st.write("Load a `.safetensors` file, modify the datatypes of its weights (grouped by layer), and save a new file.")

# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if 'available_dtypes' not in st.session_state:
    # We pre-populate this list in session_state so it NEVER resets when interacting with the table.
    st.session_state.available_dtypes = [
        "float32", "float16", "bfloat16", "float64",
        "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
        "int8", "uint8", "int16", "int32", "int64", "bool"
    ]


# --- Helper Functions ---
def get_layer_pattern(key):
    # Replaces layer numbers (e.g., .0., .12.) with an '{i}' placeholder
    return re.sub(r'(?<=\.)\d+(?=\.|$)', '{i}', key)


def dtype_to_str(dtype):
    return str(dtype).replace("torch.", "")


def str_to_dtype(dtype_str):
    # Dynamically maps the string back to a torch datatype
    if hasattr(torch, dtype_str):
        return getattr(torch, dtype_str)
    return torch.float32


# --- UI for File Paths ---
col1, col2 = st.columns(2)
with col1:
    input_path = st.text_input("Local input `.safetensors` path:", placeholder="/path/to/model.safetensors")
with col2:
    output_path = st.text_input("Local output `.safetensors` path:", placeholder="/path/to/model_modified.safetensors")

# --- Loading Logic ---
if st.button("Load Weights"):
    if not os.path.exists(input_path):
        st.error("Input file does not exist. Please check the path.")
    else:
        try:
            groups = {}
            with safe_open(input_path, framework="pt", device="cpu") as f:
                # Read metadata to inform the user
                metadata = f.metadata()
                if metadata:
                    st.info(f"Found embedded metadata with {len(metadata)} keys. It will be preserved.")

                keys = f.keys()
                for key in keys:
                    pattern = get_layer_pattern(key)
                    if pattern not in groups:
                        t = f.get_tensor(key)
                        dt_str = dtype_to_str(t.dtype)

                        # Get Tensor Shape and Size in MB
                        shape_str = str(list(t.shape))
                        try:
                            # Calculate MB size based on number of elements and bytes per element
                            size_mb = (t.numel() * t.element_size()) / (1024 * 1024)
                        except Exception:
                            size_mb = 0.0  # Fallback if element_size() fails on experimental dtypes

                        # Failsafe: If a completely unknown datatype is found, append it to session state
                        if dt_str not in st.session_state.available_dtypes:
                            st.session_state.available_dtypes.append(dt_str)

                        groups[pattern] = {
                            "Pattern": pattern,
                            "Shape": shape_str,
                            "Size (MB)": round(size_mb, 2),
                            "Count": 1,
                            "Original Dtype": dt_str,
                            "Target Dtype": dt_str
                        }
                    else:
                        groups[pattern]["Count"] += 1

            st.session_state.df = pd.DataFrame(list(groups.values()))
            st.success(f"Successfully loaded {len(keys)} tensors, grouped into {len(groups)} rows.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# --- Editing and Saving Logic ---
if not st.session_state.df.empty:
    st.write("### Weight Groups")

    # Display editable dataframe
    edited_df = st.data_editor(
        st.session_state.df,
        column_config={
            "Pattern": st.column_config.TextColumn("Weight / Layer Pattern", disabled=True),
            "Shape": st.column_config.TextColumn("Tensor Shape", disabled=True),
            "Size (MB)": st.column_config.NumberColumn("Size per Tensor (MB)", disabled=True, format="%.2f"),
            "Count": st.column_config.NumberColumn("Layers Found", disabled=True),
            "Original Dtype": st.column_config.TextColumn("Original Dtype", disabled=True),
            "Target Dtype": st.column_config.SelectboxColumn(
                "Target Dtype",
                help="Select the new datatype for all tensors in this group.",
                options=st.session_state.available_dtypes,
                required=True
            )
        },
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "⚠️ **Memory Warning:** Saving will load the entire modified model into your system RAM before writing to disk.")

    if st.button("Apply Conversions & Save", type="primary"):
        if not output_path:
            st.error("Please specify an output path.")
        elif os.path.abspath(input_path) == os.path.abspath(output_path):
            st.error(
                "Input and Output paths cannot be the exact same! Please use a different name to prevent corruption.")
        else:
            try:
                pattern_to_dtype = dict(zip(edited_df["Pattern"], edited_df["Target Dtype"]))
                tensors_to_save = {}
                file_metadata = None

                with safe_open(input_path, framework="pt", device="cpu") as f:
                    # COPY METADATA EXACTLY AS IT IS
                    file_metadata = f.metadata()

                    keys = f.keys()
                    progress_text = "Reading and converting tensors..."
                    my_bar = st.progress(0, text=progress_text)

                    for idx, key in enumerate(keys):
                        pattern = get_layer_pattern(key)
                        target_dtype_str = pattern_to_dtype.get(pattern, "float32")
                        target_dtype = str_to_dtype(target_dtype_str)

                        tensor = f.get_tensor(key)

                        if tensor.dtype != target_dtype:
                            tensor = tensor.to(target_dtype)

                        tensors_to_save[key] = tensor
                        my_bar.progress((idx + 1) / len(keys), text=f"Converting: {key}")

                with st.spinner("Writing to disk (including metadata)..."):
                    # PASS THE METADATA INTO THE SAVE FUNCTION
                    save_file(tensors_to_save, output_path, metadata=file_metadata)

                st.success(f"Successfully saved new safetensors file to `{output_path}` (Metadata preserved)!")
                del tensors_to_save

            except Exception as e:
                st.error(f"An error occurred during save: {e}")

# streamlit run convert.py