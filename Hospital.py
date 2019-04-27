# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:48:33 2019

@author: abhis
"""

import pandas as pd
import numpy as np
import numpy.matlib as npm
import pulp
from pulp import *

filename = 'Hospital_data_20170426 June 26 2017 as of July 8  2018_latest_2.xlsm'
#hosp = pd.read_excel(filename,sheet_name = 'Daywise surgery matrix',usecols = "A:AZ", userows = "1:54")
df1 = pd.read_excel(filename,sheet_name = 'Daywise surgery matrix',usecols = "A:AZ",userows="1:54")
#df1 = hosp.set_index("Group", drop = False)
df2 = df1.head(53)
df2.set_index('Group', inplace=True)
#print(df2.index.values)
#print(df2.loc[89,75])
#for i in df2.index.values:
#    for j in 
maxm = 0.0
max_dict = {}
for i,j in df2.iterrows():
    print("surgeon: ",i)
    maxm = 0.0
    for anes,freq in j.items():   
        if freq > maxm:
            maxm = freq
            max_dict[i] = [anes,freq]

print(max_dict)

mat_dict = np.matlib.eye(len(max_dict))

mat_weight = np.matlib.zeros([len(max_dict),len(max_dict)])

mat_assign_1 = np.identity(len(max_dict))
#Contrained function
mat_assign_2 = np.identity(len(max_dict))

l1=list(range(1,len(max_dict)+1))

l2=list(range(1,len(max_dict)+1))

#Create binary constraint
assign_vars = pulp.LpVariable.dicts("Assign",[(i,j) for i in l1 for j in l2],0,1,LpBinary)
print("Assign Vars",assign_vars)

con_list = np.ones(len(max_dict))
    
print(con_list)
df3 = pd.read_excel(filename,sheet_name = 'CompMat',usecols = "A:AZ",userows="1:54")
df4 = df3.head(53)
df4.set_index('Group', inplace=True)

a1,a2=0,0

for i,j in df4.iterrows():
    if i not in max_dict.keys():
        continue
    else:
        for key,values in max_dict.items():
            mat_weight[a1,a2] = df4.loc[i,values[0]]
            a2=a2+1
        a1=a1+1
        a2=0

print(mat_weight)
print(mat_assign_1)
print(mat_assign_2)

#Create variable to handle variables
prob = LpProblem("Surgery Assignment",LpMaximize)

mat_assign_unop = np.multiply(mat_weight,mat_assign_1)
print(mat_assign_unop)
print(np.matrix.sum(mat_assign_unop))
            

#contraint array
#Row constraint
row_con = mat_assign_2.sum(axis=1) 

#Objective
l3=mat_weight.tolist()
#for i in np.multiply(mat_weight,con_list):
#    l3.append(i)  
prob += lpSum([(l3[i-1][j-1] * assign_vars[(i,j)]) for i in l1 for j in l2])

#Constraints
for j in l2:
    prob += lpSum(assign_vars[(i, j)] for i in l1) == 1
for i in l1:
    prob += lpSum(assign_vars[(i, j)] for j in l2) == 1
    

prob.solve()
op_sum=0
print(l1)
print(l2)
print(l3)
for i in l1:
    for j in l2:
        print(assign_vars[(i, j)].varValue)
        print(l3[i-1][j-1])
        op_sum=op_sum+ (l3[i-1][j-1]*assign_vars[(i, j)].varValue)
print(op_sum)
#Objective function
#print(np.matrix.sum(np.multiply(mat_weight,mat_assign_2)))

            
        

    