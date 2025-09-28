import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv("cancer.csv")

y=df["diagnosis"]
x=df.drop(columns=["diagnosis","id"], axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=22)

rf=RandomForestClassifier(n_estimators=400,max_depth=10)
model=rf.fit(x_train,y_train)

a=list(df.iloc[65])
b=a[2:]

print("tahmin: ",model.predict([b]))
print("skor: ",model.score(x_test,y_test))