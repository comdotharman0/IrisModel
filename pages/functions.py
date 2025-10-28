import streamlit as st 
from sklearn.base import is_regressor
import pandas as pd
import sklearn.linear_model as sl
#from regressors import return_models
import os 

class FunctionsForDiabetesModel:
  #  def __init__(self,x,regressors,package_name,modelss,_tabname,X_train,X_test,y_train,y_test,r2_score,mean_squared_error,df,func,chartsselection):
    def __init__(self,x,regressors,package_name,_tabname,X_train,X_test,y_train,y_test,r2_score,mean_squared_error,mean_absolute_error,df,chartsselection):
        self.x=x
        self.regressors=regressors
        self.package_name=package_name
        self.modelss=self.ReturnModels()
        self._tabname=_tabname
        self.X_train=X_train
        self.X_test = X_test
        self.y_train=y_train
        self.y_test=y_test
        self.r2_score=r2_score
        self.mean_squared_error=mean_squared_error
        self.mean_absolute_error=mean_absolute_error
        self.df=df
        #self.func=func
        self.chartsselection = chartsselection
        self.selected_model= sl.LinearRegression()
        
    #@st.cache_data
    def ReturnModels(self):
       for i in self.x:
          #st.write(i)
          try:
              if(is_regressor(eval(f"{self.package_name}.{i}()")) and (i[:5]!="Multi" and i!="QuantileRegressor")):
                self.regressors.append(f"{self.package_name}.{i}")
                #st.write(self.regressors)
          except BaseException as e:
              #st.write("Error: "+str(e))
              pass
       return self.regressors
    def RunModels(self,modelstr):
      model = eval(f"{modelstr}()")
      model.fit(self.X_train,self.y_train)
      pred = model.predict(self.X_test)
      mse = self.mean_squared_error(self.y_test,pred)
      mae = self.mean_absolute_error(self.y_test,pred)
      r2_scoring = self .r2_score(self.y_test,pred)
      return [model,modelstr,mse,mae,r2_scoring,pred]
    def SelectModels(self,*,tabname):
      #st.write(modelss)
      model1 = st.session_state.my_models
      modelreal=""
      #modelreal = sl.LinearRegression()
      for i in self.modelss:
          #st.write(i)
          if model1==i:
                  modelreal = self.RunModels(i)
                  tabname.header(f"Model Selected : {i}")
                  break
    #st.header(type(modelreal))
    #modelreal.fit(X_train,y_train)
      pred= modelreal[5]
      self.selected_model=modelreal[0]
      st.header("self.Selected_model: "+str(self.selected_model))
      data = pd.DataFrame({"Actual": self.y_test,"Predicted":pred,"r2_score":self.r2_score(self.y_test,pred)*len(pred),"MSE":self.mean_squared_error(self.y_test,pred)})
      tabname.dataframe(data)
      return modelreal
    def MakePredictions(self,*,values,tabname):
        self.selected_model.fit(self.X_train,self.y_train)
        predss=self.selected_model.predict([values])
        tabname.header("The predicted Value is :"+str(predss[0])) 
        tabname.header("All Models Predictions:")
        tabname.dataframe(pd.DataFrame({"Models":[self.RunModels(i)[1] for i in self.modelss],
                                       "Predictions":[self.RunModels(i)[0].predict([values])[0] for i in self.modelss]}))
        return predss
        
        

def SelectModels(*modelss,_tabname,X_train,y_train,y_test,r2_score,mean_squared_error):
    #st.write(modelss)
    model1 = st.session_state.my_models
    modelreal=""
    #modelreal = sl.LinearRegression()
    for i in modelss:
        #st.write(i)
        if model1==i:
                modelreal = eval(f"{i}()")
                _tabname.header(f"Model Selected : {i}")
                break
    #st.header(type(modelreal))
    modelreal.fit(X_train,y_train)
    pred= modelreal.predict(X_test)
    data = pd.DataFrame({"Actual": y_test,"Predicted":pred,"r2_score":r2_score(y_test,pred)*len(pred),"MSE":mean_squared_error(y_test,pred)})
    _tabname.dataframe(data)
    return [modelreal,pred,data]
def DrawCharts(df,func):
    #st.dataframe(df)
    for i in df:
        for j in df:
            if(i==j):
                func(df,x=i,y=j,color="#ffaa00")
                #plt.xlabel(i)
                #plt.ylabel(j)
                #st.pyplot()
#@st.cache_data                
def SelectCharts(*,df,**chartsselection) :
    st.header(type(chartsselection))
    st.dataframe(chartsselection)
    chartss = st.session_state.my_charts_type
    for i in chartsselection["chartsselection"]:
      #try:
      if i==chartss:
        st.header(i)
        #DrawCharts(df,chartsselection["chartsselection"][i])
        break 
      #except BaseException as e:
       #     st.write(e)
          
def ModelsRun(modelstr):
    model = eval(f"{modelstr}()")
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    mae = mean_absolute_error(y_test,pred)
    r2_scoring = r2_score(y_test,pred)
    return [model,modelstr,mse,mae,r2_scoring,pred]
  