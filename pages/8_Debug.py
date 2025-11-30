import streamlit as st
import sys
from utils.data_loader import connect_drive

def get_drive_file_size(file_id):
    drive = connect_drive()
    file = drive.files().get(fileId=file_id, fields='name, size').execute()
    size_mb = int(file.get('size', 0)) / (1024**2)
    return f"{file['name']} — {size_mb:.2f} MB"

st.set_page_config(page_title='Cache Memory Debug', page_icon='🧠', layout='wide')
st.title('🧠 Streamlit Cache / Session Memory Usage')

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}T{suffix}"

total_size = 0
if st.session_state:
    st.subheader('🔍 Cached Objects:')
    for k, v in st.session_state.items():
        try:
            size = sys.getsizeof(v)
        except Exception:
            size = 0
        st.write(f"**{k}** → {sizeof_fmt(size)}")
        total_size += size
else:
    st.info('No items currently in session_state.')

st.subheader("📦 Google Drive File Sizes")
for key in ['PARKING_ENCODER_ID', 'ROUTES_DATA_ID', 'PARKING_VIS_CHATBOT_ID', 'PARKING_CHATBOT_ID', 'PARKING_DATA_ID']:
    if key in st.secrets:
        st.write(get_drive_file_size(st.secrets[key]))

st.markdown('---')
st.subheader('💾 Total Approximate Memory')
st.write(sizeof_fmt(total_size))
st.caption("Estimates Python object sizes, not total system memory.")
