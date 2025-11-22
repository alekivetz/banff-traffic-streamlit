from PIL import Image
import streamlit as st
import os

def display_image(file_name):
    script_dir = os.path.dirname(__file__)
    asset_dir = os.path.join(script_dir, '../assets')

    return Image.open(os.path.join(asset_dir, file_name))


def display_banner():
    script_dir = os.path.dirname(__file__)
    banner_path = os.path.join(script_dir, '../assets/banner.png')

    banner = Image.open(banner_path)

    # Resize banner
    target_width = 1500
    aspect_ratio = banner.height/ banner.width
    target_height = int(target_width * aspect_ratio)

    banner_resized = banner.resize((target_width, target_height))

    # Display banner
    st.image(banner_resized, width='stretch')