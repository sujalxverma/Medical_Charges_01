import pandas as ps
import numpy as ny
from sklearn.linear_model import LinearRegression
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

medical_df = ps.read_csv('data.csv')
min_age = 18
max_age = 64

# creates a histogram of age
# fig1 = px.histogram(data_frame=medical_df,x='age',marginal='box',nbins=47,color_discrete_sequence=['red'],title="histogram of ages")
# fig1.update_layout(bargap = 0.1)
# fig1.show()

#creates a histogram of charges and have a differentiating varaible of the smoker with value yes and no
# fig2 = px.histogram(medical_df, 
#                    x='charges', 
#                    marginal='box', 
#                    color='smoker', 
#                    color_discrete_sequence=['green', 'red'], 
#                    title='Annual Medical Charges')
# fig2.update_layout(bargap=0.1)
# fig2.show()

# counts values yes and no
# medical_df.smoker.value_counts()

# histogram of smokers differentiate by gender
# px.histogram(medical_df, x='smoker', color='sex', title='Smoker')

# scatter graph btw charges and smoker, differentiate by smoker
# fig3 = px.scatter(medical_df, 
#                  x='age', 
#                  y='charges', 
#                  color='smoker', 
#                  opacity=0.8, 
#                  hover_data=['sex','children','bmi'], 
#                  title='Age vs. Charges')
# fig3.update_traces(marker_size=5)
# fig3.show()

# gives the relation btw 1 and -1
# medical_df.charges.corr(medical_df.age)


def estimate_charges(ages,bmi,child,smoker_value,w1,w2,w3,w4,b):
    return (w1*ages + w2*bmi + w3*child + w4*smoker_value) + b



def try_parameters(w1,w2,w3,w4, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    bmi = non_smoker_df.bmi
    child = non_smoker_df.children
    smoker_value = non_smoker_df['smoker_numeric']
    predictions = estimate_charges(ages,bmi,child,smoker_value ,w1,w2,w3,w4, b)
    

    plt.plot(ages, predictions, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8,alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual'])
    
    loss = rmse(target, predictions)
    print("RMSE Loss: ", loss)

gender_values= {"female":0,"male":1}
medical_df['gender_numeric'] = medical_df.sex.map(gender_values)


smoker_values = {'no': 0, 'yes': 1}
medical_df['smoker_numeric'] = medical_df.smoker.map(smoker_values)

# print(medical_df.charges.corr(smoker_numeric))


# using linear regression for dataframe values for no smoker

non_smoker_df = medical_df[medical_df.smoker == 'no']
def rmse(targets, predictions):
    return ny.sqrt(ny.mean(ny.square(targets - predictions)))

# model = LinearRegression()
# inputs = non_smoker_df[['age','bmi','children','smoker_numeric']]
targets = non_smoker_df.charges
# model.fit(inputs,targets)  #after will get coef_ which is an array and intercept_ which is the intercept on y axis

# this gives the charge values of the patient at age 23,37,61 
# print(model.predict(ny.array([[23], 
#                         [37], 
#                         [61]]))) 

# predictions = model.predict(inputs)
# print(predictions)
# loss = rmse(targets,predictions)  #it will give the differnece in the real vs predict value of charges
# print(loss)
# Create inputs and targets
inputs, targets = medical_df[['age', 'bmi', 'children', 'smoker_numeric']], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)
try_parameters(model.coef_[0],model.coef_[1],model.coef_[2],model.coef_[3],model.intercept_)


plt.show()


