import streamlit as st
import sys
st.write(sys.path)
sys.path.insert(1,"pages")
page1 = st.Page("pages/iris.py", title="Iris Dataset Model", icon="🌷")
  
page2 = st.Page("pages/app.py", title="Diabetes Dataset Model", icon="🏥")

    # Create the navigation
pg = st.navigation([page1,page2])

    # Run the navigation
pg.run()
