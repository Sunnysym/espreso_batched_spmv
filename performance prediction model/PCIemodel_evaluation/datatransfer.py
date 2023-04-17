import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/12811/Desktop/simulation/transfer_model/transfermodel_16MB.csv")
data = np.array(data)
cols = data[:, 0]
htod = data[:, 1]
pre_htod = data[:,2]
dtoh = data[:,4]
pre_dtoh = data[:,5]
colname=['1','2','4','8','16']

x_num = np.arange(5)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize']=(5,2)
#COLOR GROUP
fig = plt.figure()
ax1 = fig.add_subplot(111)
# dark blue #3762AF dark orange #D76213 dark green #62983E
# deep blue #335D95 deep red #B1373E
# light dark blue #4C73AF and #C34A52
bar1 = ax1.bar(x_num, dtoh , label='observed', color='#78CDDE', width=0.2)
bar2 = ax1.bar(x_num + 0.2, pre_dtoh, label='estimated', color='#F17275', width=0.2)
#bar3 = ax1.bar(x_num + 0.4, plus_pipemax, label = 'repartition', color='#8CC068', width = 0.2)
ax1.set_xlabel('nStreams',fontsize=12)
ax1.set_ylabel('Time(ms)',fontsize=12)
#ax1.set_title('Weak scaling')
ax1.set_xticks(x_num+ 0.1, colname)
ax1.set_ylim([0, 2])
#ax1.legend()
#ax1.set_yscale('log')

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=True, edgecolor='k', ncol=5, handlelength=0.9, handleheight=0.9, fontsize=10)
#lns = [bar1,bar2,bar3,plot1,plot2]
#labels = [l.get_label() for l in lns]
#plt.legend(lns,labels)
#fig.savefig('C:/Users/12811/Desktop/result/dataset/weak.svg', bbox_inches='tight', pad_inches=0)
#plt.tight_layout()
plt.subplots_adjust(left=0.11,bottom=0.22,right=0.97,top=0.94,wspace=0.2,hspace=0.2)
plt.show()