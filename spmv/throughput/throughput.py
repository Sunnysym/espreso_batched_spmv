import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

data = pd.read_csv("4096mtx_batch_stream.csv")
data = np.array(data)
cols = data[:, 0]
onestreams = data[:, 1]
twostreams = data[:,2]
threestreams = data[:,3]
fourstreams = data[:,4]
fivestreams = data[:,5]
colname=['1','16','32','64','128','256','512','1024']

x_num = np.arange(8)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize']=(5,3)
#COLOR GROUP
fig = plt.figure()
ax1 = fig.add_subplot(111)
# dark blue #3762AF dark orange #D76213 dark green #62983E
ax1.plot(x_num, onestreams,label='1 stream', color='#FFC000',ms=5,lw=2,marker='o')
ax1.plot(x_num, twostreams,label='2 stream', color='#ED7D31',ms=5,lw=2,marker='o')
ax1.plot(x_num, threestreams,label='3 stream', color='#5B9BD5',ms=5,lw=2,marker='o')
ax1.plot(x_num, fourstreams,label='4 stream', color='#70AD47',ms=5,lw=2,marker='o')
ax1.plot(x_num, fivestreams,label='5 stream', color='#A5A5A5',ms=5,lw=2,marker='o')
ax1.set_xlabel('Batch Size',fontsize=12)
ax1.set_ylabel('GB/s',fontsize=12)
#ax1.set_title('Weak scaling')
ax1.set_xticks(x_num, colname)
#ax1.set_ylim([0, 2])
#ax1.legend()
#ax1.set_yscale('log')

fig.legend(loc='upper center', bbox_to_anchor=(0.52, 1), frameon=True, edgecolor='k', ncol=5, handlelength=0.9, handleheight=0.9, columnspacing=1.1, fontsize=10)
#lns = [bar1,bar2,bar3,plot1,plot2]
#labels = [l.get_label() for l in lns]
#plt.legend(lns,labels)
#fig.savefig('C:/Users/12811/Desktop/result/dataset/weak.svg', bbox_inches='tight', pad_inches=0)
#plt.tight_layout()
plt.subplots_adjust(left=0.11,bottom=0.14,right=0.95,top=0.87,wspace=0.2,hspace=0.2)
plt.show()