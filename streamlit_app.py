import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# Preparing the Iris Model
st.title("Iris Dataset Model")
st.header(" Machine Learning Model Prepared Using Logistic Regression For Iris Dataset")
iris = load_iris()
#print(dir(st))
X_train, X_test, y_train, y_test = train_test_split(iris["data"],iris["target"],test_size=0.2,random_state=42)
model= LogisticRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
flowertype = ["Iris-Setosa","Iris-Versicolour","Iris-Virginica"]
def IrisModel(a,b,c,d):
    return model.predict([[a,b,c,d]])

#st.write(f"{pred}==={y_test}")
#Tabs created
tables, charts, usemodel = st.tabs(["📋 Tables","📊 Charts","💻 Use Model"])
tabs_font_css = """
<style>
button[data-baseweb="tab"] {
  font-size: 56px;
}
</style>
"""

st.markdown(tabs_font_css, unsafe_allow_html=True)

#Use Model Tab Items
slidera = usemodel.slider("Sepal Length",4.3,7.9,0.1)
sliderb = usemodel.slider("Sepal Width",2.0,4.4,0.1)
sliderc = usemodel.slider("Petal Length",1.0,6.9,0.1)
sliderd = usemodel.slider("Petal Width",0.1,2.5,0.1)
usemodel.write(f"The Type of the flower is {flowertype[IrisModel(slidera,sliderb,sliderc,sliderd)[0]]}.")


#st.write(iris)
df = pd.DataFrame(iris["data"],columns=iris["feature_names"])
df["target"]= iris["target"]
tables.header("Iris Dataset",divider="orange")
tables.dataframe(df)
#st.write(dir(charts))
# DrawPieChart Function to draw pie charts
def DrawPieChart(tabname,data,labels,explode=None):
    fig, ax = plt.subplots()
    ax.pie(data,explode=explode,labels=list(labels),autopct="%1.1f%%",shadow=True,startangle=90)
    ax.axis("equal")
    tabname.pyplot(fig)

DrawPieChart(charts,[0,1,2],flowertype) 
#Tables Tab Items
tables.header("Correlation Matrix",divider="blue")
tables.table(df.corr())
tables.header("Descriptive Statistics of the Iris Dataset",divider="blue")
tables.table(df.describe())
#Charts Tab Items
charts.header("Iris Dataset Visualization",divider="blue")
#DrawGraph Function for drawing graphs
def DrawGraph(df,tabname,func,non_interact=None,*args,**kwargs):
    for i in df:
        for j in df:
            colors = np.random.choice(["#0000FF","#008400","#FFE747","#FF7E47","#9A00E2"])
            charts.write(colors)
            if(i!=j and i!=non_interact and j!=non_interact):
                tabname.header(f"{i} vs {j}")
                tabname.caption(f"Relationship between {i} and {j}")
               # func(*args,**kwargs)
                func(df,x=i,y=j,color=colors)
#DrawGraph(df,charts,charts.scatter_chart,"target",(df))
for i in df:
    charts.metric("Line",i,chart_data=df[i],chart_type="area",delta=f"{df[i][len(df[i])-1]-df[i][0]}",border=True)
#st.write(dir(st))

def Change_Chart_Type():
    
    chart_type1 = st.session_state.my_chart_type
    if chart_type1==None:
        DrawGraph(df,charts,charts.line_chart,"target",(df))
    match chart_type1:
        case "Line":
            DrawGraph(df,charts,charts.line_chart,"target",(df))
        case "Bar":
            DrawGraph(df,charts,charts.bar_chart,"target",(df))
        case "Area":
            DrawGraph(df,charts,charts.area_chart,"target",(df))
        case "Scatter":
            DrawGraph(df,charts,charts.scatter_chart,"target",(df))
        case "Vega Lite":
            DrawGraph(df,charts,charts.vega_lite_chart,"target",(df))
        case "Altair":
            DrawGraph(df,charts,charts.altair_chart,"target",(df))
        case _:
            DrawGraph(df,charts,charts.line_chart,"target",(df))
charts.header("Draw Iris Dataset Charts")          
chart_type = charts.radio(
    'Choose chart type',
    ["Line", "Bar","Area","Scatter"],
    key="my_chart_type",
    index=None,
    # Default to Light
    horizontal=False ,
    on_change= Change_Chart_Type# Display options horizontally
)

page1 = st.Page("pages/page1.py", title="Page One", icon=":material/home:")
    #page2 = st.Page("page_2_content.py", title="Page Two", icon=":material/looks_two:")

    # Create the navigation
pg = st.navigation([page1])

    # Run the navigation
pg.run()
