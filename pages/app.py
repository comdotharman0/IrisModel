import numpy as np
import warnings
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from diabetes.functions import SelectModels,DrawCharts,SelectCharts,ModelsRun,FunctionsForDiabetesModel
from sklearn.datasets import load_diabetes
import sklearn.linear_model as sl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.base import is_regressor
warnings.filterwarnings("ignore")
diab = load_diabetes()
X_train,X_test,y_train,y_test = train_test_split(diab.data,diab.target,test_size=0.2,random_state=42)



#return_models = funcs.ReturnModels()
#st.write(type(return_models))





#@st.cache_resource




tables,charts,modelsselection,datasummary = st.tabs(["Tables","Charts","Models","Data Summary"])  

with tables:
    st.header("Diabetes Dataset")
    df = pd.DataFrame(diab.data, columns=diab["feature_names"])
    df["target"]= diab.target
    st.dataframe(df)
    st.header("Correlation Matrix")
    st.dataframe(df.corr())
    st.header("Descriptive Statistics of the Diabetes Dataset")
    st.dataframe(df.describe())
    st.dataframe(df.info())
    st.header("Missing Values")
    st.dataframe(df.isnull().sum())
chartsselection = {"Line Chart":charts.line_chart,"Bar Chart":charts.bar_chart,"Area Chart": charts.area_chart,"Scatter Chart":charts.scatter_chart,"Vega Lite Chart":charts.vega_lite_chart,
                   #"Altair Chart":st.altair_chart
                  }


#@st.cache_data

with charts:
    st.header("Charts")
    funcs = FunctionsForDiabetesModel(dir(sl),[],"sl",modelsselection,X_train,X_test,y_train,y_test,r2_score,mean_squared_error,mean_absolute_error,pd.DataFrame(diab.data),chartsselection)
    #st.write(funcs.modelss)  
   #st.dataframe(df)
    chartss= st.selectbox('Select a Chart',
                          chartsselection.keys(),
                          key="my_charts_type",
                          on_change=SelectCharts,
                          #args=(df),
                          kwargs={"df":df,"chartsselection":chartsselection})
with modelsselection:
    models = st.selectbox("Select a Model",funcs.modelss,key="my_models",on_change=funcs.SelectModels, 
           #args=(modelsselection)
                          kwargs={"tabname": modelsselection}
                         ) 
    
    st.header("Model Selected: "+str(models))
    modelmetrics = st.expander("Model Metrics")
    predsss=st.expander("Make a Prediction")
    values = []
    for i in diab.feature_names:
        
        sliderss=predsss.slider(i.capitalize(),min_value=df[i].min(),max_value=df[i].max(),key=i,on_change=funcs.MakePredictions,
                                #args(values),
                                kwargs={"values":values,"tabname":predsss})
        values.append(sliderss)
          

return_models=funcs.ReturnModels()





data = pd.DataFrame([funcs.RunModels(i) for i in funcs.modelss],columns=["Model","ModelName","MSE","MAE","R2_Score","Predictions"])
data["Model"] = data["Model"].astype(str)
col1,col2 = modelsselection.columns(2)

    
#st.dataframe(data2)
badges = []
#st.write(dir(st.column_config))
for i in data["R2_Score"]:
    if(i==data["R2_Score"].max()):
        badges.append("🏆 Best Model with Best Score")
    elif(i<0):
        badges.append("📉 Need So Much Improvement")
    else:
        badges.append("👍 Good Model")
data["Classification"] = badges
data["Prediction Results"]=data["Predictions"]
col2.header("All Models Comparison")
col2.dataframe(data,
              column_config={"Classification":st.column_config.MultiselectColumn(
                  "Classification",
                  help="Overall description of the Model",
                  options=[
                      "🏆 Best Model with Best Score",
                      "📉 Need So Much Improvement",
                      "👍 Good Model"],
                  color=["#33CC33","#EB3838","#0F4995"],
                 # color=["green","red","blue"],
                  format_func=lambda x: x.capitalize(),),
    "Predictions":st.column_config.AreaChartColumn("Graph Predictions",y_min=0,y_max=1000,help="Graphical Representations of Predictions",),})
#st.header(type(data["Prediction Results"]))
for i in range(len(funcs.modelss)):
    #for j in data:
    modelmetrics.metric(data["ModelName"][i],data["ModelName"][i],delta=data["R2_Score"][i],chart_data=data["Prediction Results"][i],chart_type="area",border=True)
with datasummary:
    st.header("Data Summary")
    st.write(diab["DESCR"])
#MyBadge("Jai Jai Siya Ram!","https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1HPeOV3r2Kff0d1IzQ82u2IUjThHS-4cp6uIUJibBTw&s=10")
#st.header(st.badge("Hi Hlo JAI JAI SIYA RAM !",icon="📊",width=500))
