import streamlit as st
page1 = st.Page("pages/iris.py", title="Iris Dataset Model", icon="🌷")
  
page2 = st.Page("pages/diabetes.py", title="Diabetes Dataset Model", icon="🏥")

    # Create the navigation
pg = st.navigation([page1,page2])

    # Run the navigation
pg.run()
