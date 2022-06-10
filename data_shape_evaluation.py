#test runtime performance under different data shape

import tensorflow as tf
import numpy as np
import os
from pycoral.utils import edgetpu
import time
import matplotlib.pyplot as plt
import copy
from tflite import tfmodel
import decompose1 as dm


# base multiplier
#k: bitwidth of multiplier;input range:[0, 2^k]
class uint_k(tf.keras.layers.Layer):
    def __init__(self,k):
        super().__init__()
        self.k=k
    def call(self, input1,input2):
        tf.debugging.assert_less(input1,tf.constant(2**self.k,dtype=tf.float32))
        tf.debugging.assert_less(input2,tf.constant(2**self.k,dtype=tf.float32))
        return tf.matmul(input1, input2)
        
# Karatsuba multiplier using base multiplier
# shape1, shape2: input data shape; bitwidth: a list of pre-defined the bitwidth base mutiplier. For example, the list should be [4,8] if you need uint8 and uint4 multiplier.
# input1,input2: input data
# t1: allocation time + computation time
# t2: computation time
# return opt which is the multiplication output

class uint_k_base():
    def __init__(self,shape1,shape2,bitwidth):
        base_model={}
        self.bitwidth=max(bitwidth)
        self.bit=2**self.bitwidth
        for i in bitwidth:
            input_1 = tf.keras.layers.Input(shape=shape1)
            input_2 = tf.keras.layers.Input(shape=shape2)
            output = uint_k(i)(input_1,input_2)
            full_model = tf.keras.Model(inputs=(input_1,input_2), outputs=(output))
            model_class=tfmodel(full_model);
            base_model[i]=(model_class,model_class.quantize('uint_%s' %i,i))
        self.base=base_model
    def call(self,input1,input2,t1,t2):
        start=time.perf_counter()
        assert(input1.shape[0]==input2.shape[0])
        opt=[]
        length=int(np.ceil(np.log(max(input1.max(),input2.max())+1)/np.log(self.bit)))
        length=2**int(np.ceil(np.log2(length)))
        opt_tmp,t1,t2=karatsuba(dm.decompose_input(input1,length,self.bitwidth),dm.decompose_input(input2,length,self.bitwidth),self.base,t1,t2,self.bitwidth)
        opt=dm.combine(opt_tmp,self.bitwidth)
        end=time.perf_counter()
        return opt,(end-start),t1,t2
        
# A,B input data
# base multiplier interpreter
# t1: allocation time + computation time
# t2: computation time
# bitwidth: karatsuba decompose bitwidth
        
def karatsuba(A,B,base,t1,t2,bitwidth=8):
    base_bit=2**bitwidth
    assert(len(A.shape)==len(B.shape)==4)
    assert(A.shape[0]==B.shape[0])
    assert(A.shape[1]==B.shape[1])
    batch=A.shape[1]
    n=A.shape[0]
    R= np.zeros((2*n-1,batch,A.shape[2],B.shape[3]), dtype =int)
    if n==1:
        for i in range(batch):
            ipt1=np.expand_dims(A[0][i], axis=0);ipt2=np.expand_dims(B[0][i], axis=0)
            max_value=max(ipt1.max(),ipt2.max())
            if max_value<base_bit:
                start=time.perf_counter()
                bit_base=base[min([i for i in base.keys() if 2**i>max_value])]
                interpreter=bit_base[1][2]
                R[0][i],t_tmp=bit_base[0].evaluate_tflite_model(interpreter,ipt1,ipt2,1)
                end=time.perf_counter()
                t2+=t_tmp
                t1+=end-start
            else:
                length=int(np.ceil(np.log(max(ipt1.max(),ipt2.max())+1)/np.log(base_bit)))
                length=2**int(np.ceil(np.log2(length)))
                R_tmp,t1,t2=karatsuba(dm.decompose_input(ipt1,length,bitwidth),dm.decompose_input(ipt2,length,bitwidth),base,t1,t2,bitwidth)
                R[0][i]=dm.combine(R_tmp,bitwidth)
        return np.array([R[0]]),t1,t2
    D1 = np.array(A[:n//2])
    D0 = np.array(A[n//2:])
    E1 = np.array(B[:n//2])
    E0 = np.array(B[n//2:])
    P,t1,t2 = karatsuba(D1,E1,base,t1,t2,bitwidth)
    T,t1,t2 = karatsuba(D0,E0,base,t1,t2,bitwidth)
    X,t1,t2 = karatsuba(D1+D0,E1+E0,base,t1,t2,bitwidth)
    M = X-P-T
    R[0:n-1]+=P
    R[n:2*n-1] +=T
    R[(n // 2):n - 1 + (n // 2)] += M
    return R,t1,t2
    
    
low=0;high=2**16-1;
time1=[]
time2=[]
time3=[]
size_list=[16,32,64,128,256]
for size in size_list:
    input1=np.random.randint(low, high, size=[128,size,size], dtype=int)
    input2=np.random.randint(low, high, size=[128,size,size], dtype=int)
    model_class=uint_k_base(tuple(input1.shape[1:]),tuple(input2.shape[1:]),[2,3,4,5,6,7,8])
    opt,ttt,t1,t2=model_class.call(input1,input2,0,0)
    time1.append(ttt)
    time2.append(t1)
    time3.append(t2)
    print(size,ttt,t1,t2)
    # print(opt)
    # print(np.matmul(input1,input2))
    # print((opt-np.matmul(input1,input2))/np.matmul(input1,input2))
    
plt.plot(size_list, time1)
plt.plot(size_list, time2)
plt.plot(size_list, time3)
plt.xscale('log',base=2)
plt.yscale('log',base=2)
plt.show()