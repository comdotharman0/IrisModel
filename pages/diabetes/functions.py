import streamlit as st 
from sklearn.base import is_regressor
import pandas as pd
import sklearn.linear_model as sl
#from regressors import return_models
import time

class FunctionsForDiabetesModel:
  #  def __init__(self,x,regressors,package_name,modelss,_tabname,X_train,X_test,y_train,y_test,r2_score,mean_squared_error,df,func,chartsselection):
    def __init__(self,x,regressors,package_name,_tabname,X_train,X_test,y_train,y_test,r2_score,mean_squared_error,mean_absolute_error,df,chartsselection,tabs):
        self.x=x
        self.regressors=regressors
        self.package_name=package_name
        self.modelss=self.ReturnModels()
        #st.header(type(self.modelss))
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
        self.tabs =tabs
        self._selected_model= sl.LinearRegression()



    def set_selected_model(self,model):
        self._selected_model = model

  


  
    #@st.cache_data
    def ReturnModels(self):
       
       for i in self.x:
          #st.write(i)
          try:
              if(is_regressor(eval(f"{self.package_name}.{i}()")) and  (i[:5]!="Multi" and i!="QuantileRegressor")):
                
                self.regressors.append(f"{self.package_name}.{i}")
                #st.write(self.regressors)
          except BaseException as e:
              #st.write("Error: "+str(e))
              pass
       return self.regressors





  
    def RunModels(self,modelstr):
      try:
        model = eval(f"{modelstr}()")
        #st.header(model)
        model.fit(self.X_train,self.y_train)
        pred = model.predict(self.X_test)
        mse = self.mean_squared_error(self.y_test,pred)
        mae = self.mean_absolute_error(self.y_test,pred)
        r2_scoring = self .r2_score(self.y_test,pred)
        return [model,modelstr,mse,mae,r2_scoring,pred]
      except BaseException as e:
          st.write(e)




  
    def SelectModels(self):
      try:
        model1 = st.session_state.my_models
        
        modelreal=[]
        with self.tabs["SelectModels"]:
          for i in self.modelss:
            if model1==i:
                modelreal= self.RunModels(str(model1))
                #st.header(self._selected_model)
                self.set_selected_model(modelreal[0])
                st.header(self._selected_model)
                st.header(f"Model Selected : {modelreal[1]}")
                break
          pred= modelreal[5]
          #st.header("Model Predicted")
          #st.write(modelreal[0])
          data = pd.DataFrame({"Actual": self.y_test,"Predicted":pred,"r2_score":[self.r2_score(self.y_test,pred)]*len(pred),"MSE":self.mean_squared_error(self.y_test,pred)})
          data = data if data.isnull().sum().sum()==0 else data.dropna()
          st.dataframe(data)
        return modelreal
      except BaseException as e:
        st.write(e)




  
        
    def MakePredictions(self,*,values,tabname):
        #st.header(self._selected_model)
        self._selected_model.fit(self.X_train,self.y_train)
        #st.header(self._selected_model)
        
        
        predss=self._selected_model.predict([values])
        tabname.header("The predicted Value is :"+str(predss[0])) 
        tabname.header("All Models Predictions:")
        tabname.dataframe(pd.DataFrame({"Models":[self.RunModels(i)[1] for i in self.modelss],
                                       "Predictions":[self.RunModels(i)[0].predict([values])[0] for i in self.modelss]}))
        return predss



  
    def DrawCharts(self,df,func,i,j):
      #st.header(self.modelss)
      try:
        func(df,x=i,y=j,color="#ffaa00")
      except BaseException as e:
        st.code(str(e))
                #plt.xlabel(i)
                #plt.ylabel(j)
                #st.pyplot()




  
    def SelectCharts(self,df,**chartsselection) :
      #try:
        chartss = st.session_state.my_charts_type
        chartx = st.session_state.ChartX
        charty = st.session_state.ChartY
        for i in chartsselection["chartsselection"]:
      #try:
          if i==chartss:
            st.header(i)
            #with self.tabs["SelectCharts"]:
          self.DrawCharts(df,chartsselection["chartsselection"][i],chartx,charty)
          break 
      #except BaseException as e:
            #st.write(str(e))
    
          

        
        

