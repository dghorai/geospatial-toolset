Streamlit's Internal Config Location: C:\Users\USER\.streamlit

Temporary File Location (Uploaded Files): 
- When you use st.file_uploader, the files are stored in RAM (Memory), not on your hard drive.
- If you need to find a physical "location" for an uploaded file to pass to a tool that requires a path, you must create a Temporary Directory yourself using Python:

<pre>
import tempfile
import os

if uploaded_file:
    # 1. Create a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 2. Create a path inside that directory
        path = os.path.join(tmp_dir, uploaded_file.name)
        
        # 3. Write the content to that path
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # 4. Now you have a 'full path' for your code to use
        st.write(f"Internal Server Path: {path}")
        gdf = gpd.read_file(path)
</pre>