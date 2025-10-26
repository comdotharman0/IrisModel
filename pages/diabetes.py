import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import sklearn.linear_model as sl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.base import is_regressor
diab = load_diabetes()
X_train,X_test,y_train,y_test = train_test_split(diab.data,diab.target,test_size=0.2,random_state=42)




def ReturnModels(x,regressors,package_name):
    for i in x:
        try:
            if(is_regressor(eval(f"{package_name}.{i}()")) and (i[:5]!="Multi" and i!="QuantileRegressor")):
                regressors.append(f"{package_name}.{i}")
        except BaseException:
            pass
    return regressors
return_models = ReturnModels(dir(sl),[],"sl")
#st.write(type(return_models))






def SelectModels(*modelss,tabname):
    #st.write(modelss)
    model1 = st.session_state.my_models
    modelreal=""
    #modelreal = sl.LinearRegression()
    for i in modelss:
        #st.write(i)
        if model1==i:
                modelreal = eval(f"{i}()")
                tabname.header(f"Model Selected : {i}")
                break
    #st.header(type(modelreal))
    modelreal.fit(X_train,y_train)
    pred= modelreal.predict(X_test)
    data = pd.DataFrame({"Actual": y_test,"Predicted":pred,"r2_score":r2_score(y_test,pred)*len(pred),"MSE":mean_squared_error(y_test,pred)})
    tabname.dataframe(data)
    return [modelreal,pred,data]




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
chartsselection = {"Line Chart":st.line_chart,"Bar Chart":st.bar_chart,"Area Chart": st.area_chart,"Scatter Chart":st.scatter_chart,"Vega Lite Chart":st.vega_lite_chart,
                   #"Altair Chart":st.altair_chart
                  }
def DrawCharts(df,func):
    for i in df:
        for j in df:
            if(i!=j):
                func(df,x=i,y=j,color="#ffaa00")
                #plt.xlabel(i)
                #plt.ylabel(j)
                #st.pyplot()
                
def SelectCharts() :
    chartss = st.session_state.my_charts_type
    for i in chartsselection:
        if i==chartss:
            DrawCharts(df,chartsselection[i])
            break
with charts:
    st.header("Charts")
    chartss= st.selectbox('Select a Chart',chartsselection.keys(),key="my_charts_type",on_change=SelectCharts)
with modelsselection:
    models = st.selectbox("Select a Model",ReturnModels(dir(sl),[],"sl"),key="my_models",on_change=SelectModels, 
           args=(return_models),kwargs={"tabname": modelsselection}) 




def ModelsRun(modelstr):
    model = eval(f"{modelstr}()")
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    mae = mean_absolute_error(y_test,pred)
    r2_scoring = r2_score(y_test,pred)
    return [model,modelstr,mse,mae,r2_scoring,pred]



data = pd.DataFrame([ModelsRun(i) for i in return_models],columns=["Model","ModelName","MSE","MAE","R2_Score","Predictions"])
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
for i in range(len(return_models)):
    #for j in data:
    modelsselection.metric(data["ModelName"][i],data["ModelName"][i],delta=data["R2_Score"][i],chart_data=data["Prediction Results"][i],chart_type="area",border=True)
#MyBadge("Jai Jai Siya Ram!","https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1HPeOV3r2Kff0d1IzQ82u2IUjThHS-4cp6uIUJibBTw&s=10")
#st.header(st.badge("Hi Hlo JAI JAI SIYA RAM !",icon="📊",width=500))
