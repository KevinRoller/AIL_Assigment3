from tkinter import Y
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
df=pd.DataFrame([[-1,2,3],
                 [-2,4,4],
                 [2,4,5],
                 [3,3,8]])
npar=df.to_numpy()   
# # df.drop(index=1,axis=0,inplace=True)
# # print(df.iloc[1][:])
# # print(re.split("[, ]","adfsf,adfasdf dfadsf"))
# # test=np.array([[-1,2,3],
# # [-2,4,4],
# # [2,4,5],
# # [3,3,8]])
# # for i in test:
# #     print(i)
# # for i in range(20):
# #     print(i)
# #     i+=1
# # x=np.linspace(10,20,30)
# # y=x**2-4*x+3
# # fig1=plt.figure()
# # plt.plot(x,y)
# # fig1.show()
# # input()
# print(npar[:,1]==4)



# import matplotlib.pyplot as plt
# import numpy as np


# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]

# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x, labels)
# ax.legend()

# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)

# fig.tight_layout()

# temp=npar[:,1]
# print(type(temp[:]))
# #npar[:,1]=3
# temp[:]=3
# print(npar)
# print(temp)
#arr = np.array([1, 2, 4, 8, 16, 32])
print(npar.shape)
npar=np.concatenate((npar,np.array([[1,2,3,4]]).T),axis=1)

for i in range(1,6):
    print(i)