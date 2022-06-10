import numpy as np
import time
from tflite import tfmodel
import tensorflow as tf
from pycoral.utils import edgetpu
import numpy as np

# def combine(input,bitwidth=8):
#     base_bit=2**bitwidth
#     output=0
#     input=np.flipud(input)
#     for i in range(len(input)):
#         output+=input[i]*(base_bit**i)
#     return output

# def decompose_input(input1,input2,bitwidth=8):
#     if input1==input2==0:
#         return [0],[0]
#     def fun(input):
#         base_bit=2**bitwidth
#         digit=[]
#         while (input>0):
#             digit.append(input%base_bit)
#             input=input//base_bit
#         digit=list(reversed(digit))
#         return digit
#     hex_input1=fun(input1);hex_input2=fun(input2)
#     length=max(len(hex_input1),len(hex_input2))
#     length=int(2**np.ceil(np.log2(length)))
#     hex_input1=[0]*(length-len(hex_input1))+hex_input1;hex_input2=[0]*(length-len(hex_input2))+hex_input2
#     return hex_input1,hex_input2

def combine(input,bitwidth=8):
    base_bit=2**bitwidth
    output=np.zeros_like(input[0])
    input=np.flipud(input)
    for i in range(len(input)):
        output=output+input[i]*(base_bit**i)
    return output

def decompose_input(tensor,length,bitwidth=8):
    if (tensor==0).all():
        return np.zeros_like(tensor)
    def fun(input):
        base_bit=2**bitwidth
        digit=[]
        while (input>0).any():
            digit.append(input%base_bit)
            input=input//base_bit
        digit=list(reversed(digit))
        return digit
    hex_input=fun(tensor)
    hex_input=[np.zeros_like(tensor) for i in range((length-len(hex_input)))]+hex_input
    hex_input=np.stack(hex_input,axis=0)
    return hex_input

def karatsuba(_input,interpreter,base,t1,t2,bitwidth=8):
    base_bit=2**bitwidth
    A=_input[0]
    B=_input[1]
    assert(len(A)==len(B))
    n=len(A)
    R= np.zeros(2*n-1, dtype =int)
    if n ==1:
        if max(A[0],B[0])<base_bit:
            start=time.perf_counter()
            R[0],t_tmp=base.evaluate_tflite_model(interpreter,np.array([(A[0],B[0])]),1,t2)
            end=time.perf_counter()
            t2+=t_tmp
            t1+=end-start
        else:
            R_tmp,t1,t2=karatsuba(decompose_input(A[0],B[0]),interpreter,base,t1,t2)
            R[0]=combine(R_tmp)
        return np.array([R[0]]),t1,t2
    D1 = np.array(A[:n//2])
    D0 = np.array(A[n//2:])
    E1 = np.array(B[:n//2])
    E0 = np.array(B[n//2:])
    P,t1,t2 = karatsuba((D1,E1),interpreter,base,t1,t2)
    T,t1,t2 = karatsuba((D0,E0),interpreter,base,t1,t2)
    X,t1,t2 = karatsuba((D1+D0,E1+E0),interpreter,base,t1,t2)
    M = X-P-T
    R[0:n-1]+=P
    R[n:2*n-1] +=T
    R[(n // 2):n - 1 + (n // 2)] += M
    return R,t1,t2

def main(input,base,interpreter,high,t1,t2):
    start=time.perf_counter()
    assert(tf.math.reduce_all(tf.less(input,tf.constant(high+1))))
    opt=[]
    for i in input:
        opt_tmp,t1,t2=karatsuba(decompose_input(i[0],i[1]),interpreter,base,t1,t2)
        opt.append(combine(opt_tmp))
    end=time.perf_counter()
    return opt,(end-start),t1,t2



# def combine(input,bitwidth=8):
#     base_bit=2**bitwidth
#     output=0
#     input=np.flipud(input)
#     for i in range(len(input)):
#         output+=input[i]*(base_bit**i)
#     return output

# def decompose_input(input1,input2,bitwidth=8):
#     if input1==input2==0:
#         return [0],[0]
#     def fun(input):
#         base_bit=2**bitwidth
#         digit=[]
#         while (input>0):
#             digit.append(input%base_bit)
#             input=input//base_bit
#         digit=list(reversed(digit))
#         return digit
#     hex_input1=fun(input1);hex_input2=fun(input2)
#     length=max(len(hex_input1),len(hex_input2))
#     length=int(2**np.ceil(np.log2(length)))
#     hex_input1=[0]*(length-len(hex_input1))+hex_input1;hex_input2=[0]*(length-len(hex_input2))+hex_input2
#     return hex_input1,hex_input2

# def karatsuba(_input,edgetpu_path,base,t1,bitwidth=8):
#     base_bit=2**bitwidth
#     A=_input[0]
#     B=_input[1]
#     assert(len(A)==len(B))
#     n=len(A)
#     R= np.zeros(2*n-1, dtype =int)
#     if n ==1:
#         if max(A[0],B[0])<base_bit:
#             start=time.perf_counter()
#             if edgetpu_path:
#                 R[0]=base.evaluate_edgetpu_tflite_model(edgetpu_path,np.array([(A[0],B[0])]),1)
#             else:
#                 R[0]=base.evaluate_tflite_model(np.array([(A[0],B[0])]),1)
#             end=time.perf_counter()
#             t1+=end-start
#         else:
#             R_tmp,t1=karatsuba(decompose_input(A[0],B[0]),edgetpu_path,base,t1)
#             R[0]=combine(R_tmp)
#         return np.array([R[0]]),t1
#     D1 = np.array(A[:n//2])
#     D0 = np.array(A[n//2:])
#     E1 = np.array(B[:n//2])
#     E0 = np.array(B[n//2:])
#     P,t1 = karatsuba((D1,E1),edgetpu_path,base,t1)
#     T,t1 = karatsuba((D0,E0),edgetpu_path,base,t1)
#     X,t1 = karatsuba((D1+D0,E1+E0),edgetpu_path,base,t1)
#     M = X-P-T
#     R[0:n-1]+=P
#     R[n:2*n-1] +=T
#     R[(n // 2):n - 1 + (n // 2)] += M
#     return R,t1

# def main(input,base,edgetpu_path):
#     start=time.perf_counter()
#     assert(tf.math.reduce_all(tf.less(input,tf.constant(2**16))))
#     opt={}
#     t1=0
#     for i in input:
#         opt_tmp,t1=karatsuba(decompose_input(i[0],i[1]),edgetpu_path,base,t1)
#         opt[tuple(i)]=combine(opt_tmp)
#     end=time.perf_counter()
#     return opt,(end-start),t1

# def karatsuba1(_input,bitwidth=8):
#     base_bit=2**bitwidth
#     A=_input[0]
#     B=_input[1]
#     list=set()
#     assert(len(A)==len(B))
#     n=len(A)
#     if n ==1:
#         if max(A[0],B[0])<base_bit:
#             list.add((max(A[0],B[0]),min(A[0],B[0])))
#         else:
#             list=list.union(karatsuba1(decompose_input(A[0],B[0],bitwidth),bitwidth))
#         return list
#     D1 = np.array(A[:n//2])
#     D0 = np.array(A[n//2:])
#     E1 = np.array(B[:n//2])
#     E0 = np.array(B[n//2:])
#     list=list.union(karatsuba1((D1,E1),bitwidth),karatsuba1((D0,E0),bitwidth),karatsuba1((D1+D0,E1+E0),bitwidth))
#     return list

# def karatsuba2(_input,table,bitwidth=8):
#     base_bit=2**bitwidth
#     A=_input[0]
#     B=_input[1]
#     assert(len(A)==len(B))
#     n=len(A)
#     R= np.zeros(2*n-1, dtype =int)
#     if n ==1:
#         if max(A[0],B[0])<base_bit:
#             R[0]=table[(max(A[0],B[0]),min(A[0],B[0]))]
#         else:
#             R[0]=combine(karatsuba2(decompose_input(A[0],B[0],bitwidth),table,bitwidth),bitwidth)
#         return np.array([R[0]])
#     D1 = np.array(A[:n//2])
#     D0 = np.array(A[n//2:])
#     E1 = np.array(B[:n//2])
#     E0 = np.array(B[n//2:])
#     P = karatsuba2((D1,E1),table,bitwidth)
#     T = karatsuba2((D0,E0),table,bitwidth)
#     X = karatsuba2((D1+D0,E1+E0),table,bitwidth)
#     M = X-P-T
#     R[0:n-1]+=P
#     R[n:2*n-1] +=T
#     R[(n // 2):n - 1 + (n // 2)] += M
#     return R

# def input_sample(input,bitwidth=8):
#     output_=set()
#     for i in input:
#         output_.update(karatsuba1(decompose_input(i[0],i[1],bitwidth),bitwidth))
#     return output_

# def main(input,model_class,bitwidth=8):
#     start1=time.perf_counter()
#     output_=input_sample(input)
#     start2=time.perf_counter()
#     solution1=model_class.evaluate_tflite_model(np.array(list(output_)),np.array(list(output_)).shape[0])
#     end2=time.perf_counter()
#     assert(len(solution1)==len(output_))
#     output1={tuple(output_)[i]:int(solution1.tolist()[i][0]) for i in range(len(output_))}
#     opt1={};
#     for i in input:
#         opt1[tuple(i)]=combine(karatsuba2(decompose_input(i[0],i[1],bitwidth),output1,bitwidth),bitwidth)
#     end1=time.perf_counter()
#     return opt1,(end1-start1),(end2-start2)