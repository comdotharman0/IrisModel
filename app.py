import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris["data"],iris["target"],test_size=0.2,random_state=42)
model= LogisticRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
flowertype = ["Iris-Setosa","Iris-Versicolour","Iris-Virginica"]
def IrisModel(a,b,c,d):
    return model.predict([[a,b,c,d]])

#st.write(f"{pred}==={y_test}")
slidera = st.slider("Sepal Length",4.3,7.9,0.1)
sliderb = st.slider("Srpal Width",2.0,4.4,0.1)
sliderc = st.slider("Petal Length",1.0,6.9,0.1)
sliderd = st.slider("Petal Width",0.1,2.5,0.1)
st.write(f"The Type of the flower is {flowertype[IrisModel(slidera,sliderb,sliderc,sliderd)[0]]}.")
#st.write(iris)
