import pandas as ps
import numpy as ny
from sklearn.linear_model import LinearRegression
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


medical_df = ps.read_csv('data.csv')
# print(medical_df)
#gives the statistical data about age group of our medical dataframe
# print(medical_df.age.describe())   

min_age = 18
max_age = 64

#creates a histogram between age and charges of the patient
# fig = px.histogram(data_frame=medical_df,x='age',y='charges',nbins=max_age-min_age)  
# fig.update_layout(bargap=0.1)
# fig.show()

#creates a scatterplot between age and charges and the color differentiate tells which one smoke or not
# fig1 = px.scatter(data_frame=medical_df,x='age',y='charges',color='smoker',hover_data="sex",color_discrete_sequence=['red','green'])
# fig1.show()



#creates a dataframe of the perosn who smokes
smoker_df = medical_df[medical_df.smoker=="yes"]
# print(non_smoker_df)
actual_charges = smoker_df.charges
# print(actual_charges)

smoker_age = smoker_df.age


def estimate_charges(age,w,b):
    return (age*w)+b       # a linear equation where w is slope and b is intercept on y-axis

slope_w = 350
intercept_b = 1000   


# target_check function creates a line graph to form a linear relation between age and charges
def target_check(age,w,b):
    smoker_charges = estimate_charges(age,w,b)
    plt.scatter(smoker_age,smoker_df.charges,alpha=0.8,s=8)
    plt.plot(smoker_age,smoker_charges,'g-')


target_check(smoker_age,305.23760211,20294.12812691597)

# print(smoker_charges)
# fig3 = px.line(data_frame=smoker_df,x='age',y=smoker_charges)
# fig3.show()

#use of sklearn
inputs = smoker_df[['age']]
targets = smoker_df.charges
model = LinearRegression()
model.fit(inputs,targets)
print(f"Coefficients: {model.coef_}")  
print(f"Intercept: {model.intercept_}") 

# again use target_check
target_check(smoker_age,model.coef_,model.intercept_)





plt.show()