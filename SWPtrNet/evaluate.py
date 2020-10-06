import numpy as np
from tqdm import tqdm

x, y = [], []
with open("data/tsp20_test.txt") as f:
    for l in tqdm(f):
        inputs, outputs = l.split(' output ')
        x.append(np.array(inputs.split(), dtype=np.float32).reshape([1, -1, 2]))
        y.append(np.array(outputs.split(), dtype=np.int32).reshape(1,21))

y = np.concatenate(y, axis=0)
x = np.concatenate(x, axis=0)

print (x.shape,y.shape)

# a=np.load('data/tsp_(20,20)_test=10000.npz')
# x=a["x"][:1000]
# y=a["y"][:1000]
#
# from matplotlib import pyplot as plt
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     for j in range(20):
#         plt.scatter(x[i][j][0],x[i][j][1])
#         plt.text(x[i][j][0],x[i][j][1],str(j+1))
# plt.show()



index_list=[]
loss_list=[]
optimal_list=[]
result_list=[]
accuracy_list=[]
for index in range(100,101):
    index_list.append(index*10000)
    prefix="logs/tsp_2019-04-02_04-55-20/test_results/"+str(index)+"0000_"

    loss=0
    result=[]
    label=[]
    prob=[]
    for i in range(100):
        a=np.load(prefix + str(i+1) + ".npz")
        b=a["arr_0"][:,:21]
        result.append(b)
        label.append(a["arr_1"])
        prob.append(a["arr_2"][:,:21,:])
        loss+=a["arr_3"]

    print ("loss %f" % (loss/100.))
    loss_list.append(loss/100.)

    result=np.concatenate(result,axis=0)
    label=np.concatenate(label,axis=0)
    prob=np.concatenate(prob,axis=0)

    result_new=np.zeros(result.shape,dtype="int")
    for i in range(result.shape[0]):
        mask=np.ones(20)
        mask[0]=0
        for j in range(result.shape[1]):
            a=np.exp(prob[i][j]/100.)
            b=np.argmax(mask*a)
            result_new[i][j]=b
            mask[b]=0
    print (result)
    print (result_new)
    # print result_new

    # for i in range(1000):
    #     print "----"
    #     print result[i]
    #     print result_new[i]

    result_new=np.concatenate([np.ones([result_new.shape[0],1]),result_new[:,:-1]+1],1).astype(np.int32)
    for l in range(10):
        print (result_new[l])
        print (y[l])
    a=np.sum(np.equal(result_new,y[:,:20]),axis=1)
    accuracy_list.append(np.sum(a)/20/10000.)
    #or j in range(20):
    #   print "--------------"
    #    print result_new[j]
    #    print y[j]
    #    print np.argmax(result,2)[j]
    result=result_new
    # b=np.sum(np.equal(label[:,20],y),axis=1)
    # # c=np.sum(np.equal(result,label),axis=1)

    # print np.sum(a),np.sum(b)
    # print x.shape
    a=[]
    b=[]
    f=10000000

    for i in range(1000):
        # solver = TSPSolver.from_data(x[i,:,0]*f, x[i,:,1]*f, norm='EUC_2D')
        # solution=solver.solve()
        # q=solution.tour
        # print "----------"
        # q=[p+1 for p in q]
        # print q
        # print result[i]
        q=y[i]
        l=0
        for j in range(19):
           l+=np.sqrt((x[i][result[i][j]-1][0]-x[i][result[i][j+1]-1][0])**2+(x[i][result[i][j]-1][1]-x[i][result[i][j+1]-1][1])**2)
        l+=np.sqrt((x[i][result[i][19]-1][0]-x[i][result[i][0]-1][0])**2+(x[i][result[i][19]-1][1]-x[i][result[i][0]-1][1])**2)
        a.append(l)
        l = 0
        for j in range(19):
            l += np.sqrt((x[i][q[j] - 1][0] - x[i][q[j + 1] - 1][0]) ** 2 + (
            x[i][q[j] - 1][1] - x[i][q[j + 1] - 1][1]) ** 2)
        l += np.sqrt((x[i][q[19] - 1][0] - x[i][q[0] - 1][0]) ** 2 + (
        x[i][q[19] - 1][1] - x[i][q[0] - 1][1]) ** 2)
        b.append(l)
    print (np.mean(a))
    print (np.mean(b))
    optimal_list.append(np.mean(a))
    result_list.append(np.mean(b))

print (optimal_list)
print (result_list)
print (accuracy_list)
print (loss_list)

#optimal_gap=[(result_list[i]-optimal_list[i])/optimal_list[i] for i in range(len(optimal_list))]

#from matplotlib import pyplot as plt

#plt.subplot(1,3,1)
#plt.plot(index_list,optimal_gap)
#plt.title("optimal gap")
#plt.subplot(1,3,2)
#plt.plot(index_list,accuracy_list)
#plt.title("accuracy for true label")
#plt.subplot(1,3,3)
#plt.plot(index_list,loss_list)
#plt.title("test loss")
#plt.show()


