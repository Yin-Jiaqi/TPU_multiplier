import tensorflow as tf
import numpy as np
import os
from pycoral.utils import edgetpu
import time

class tfmodel:
    def __init__(self, model):
        self.model=model
        assert(len(model.input)==2)
        self.input_shape1=[[1]+list(model.input[0].shape)[1:]]+[[1]+list(model.input[1].shape)[1:]]
        self.input_shape2=[[1]+list(model.input[1].shape)[1:]]+[[1]+list(model.input[0].shape)[1:]]

    def quantize(self,name,k,prt=False):
        kkk=2**k
        interval=1
        class data:
            def __init__(self, shape):
                self.shape = shape
            def data_set(self):
                iter2=0
                for _ in range(kkk*int((1/interval))):
                    iter1=0;
                    for _ in range(kkk*int((1/interval))):
                        data=[]
                        for s_index in range(len(self.shape)):
                            s=self.shape[s_index]
                            if s_index==0:
                                data.append((np.ones(s)*iter1).astype(np.float32))
                            if s_index==1:
                                data.append((np.ones(s)*iter2).astype(np.float32))
                        iter1=iter1+interval
                  #data.append(np.random.randint(256, size=1).astype(np.float32))
                        yield data
                    iter2=iter2+interval
        def quantize_main(model,datas,name):
            converter1 = tf.lite.TFLiteConverter.from_keras_model(model)
            converter1.optimizations = [tf.lite.Optimize.DEFAULT]
            converter1.representative_dataset =datas.data_set
            converter1.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter1.target_spec.supported_types = [tf.int8]
            converter1.inference_input_type = tf.uint8
            converter1.inference_output_type = tf.uint8
            tflite_model1 = converter1.convert()
            with open(name+'_quant.tflite' , 'wb') as f:
                f.write(tflite_model1)
            os.system("edgetpu_compiler "+name+"_quant.tflite")
            return tflite_model1
        try:
            datas=data(self.input_shape1)
            tflite_model=quantize_main(self.model,datas,name)
        except:
            datas=data(self.input_shape2)
            tflite_model=quantize_main(self.model,datas,name)
        #print(list(datas.data_set()))
        if prt:
            print(list(datas.data_set()))
        self.tflite_model=tflite_model
        self.interpreter1=tf.lite.Interpreter(model_content=self.tflite_model)
        model_file = os.path.join(name+'_quant_edgetpu.tflite')
        self.interpreter2 = edgetpu.make_interpreter(model_file)
        return self.tflite_model,self.interpreter1,self.interpreter2
    def evaluate_tflite_model(self,interpreter,data1,data2,batch):
        # Initialize TFLite interpreter using the model.
        #import pdb;pdb.set_trace()
        #interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        assert(len(interpreter.get_input_details())==2)
        input_tensor_index1 = interpreter.get_input_details()[0]["index"]
        input_tensor_index2 = interpreter.get_input_details()[1]["index"]
        output_tensor_index = interpreter.get_output_details()[0]["index"]
        input_shape1=np.array(interpreter.get_input_details()[0]['shape'],copy=True)
        input_shape1[0]=batch
        input_shape2=np.array(interpreter.get_input_details()[1]['shape'],copy=True)
        input_shape2[0]=batch
        output_shape1=np.array(interpreter.get_output_details()[0]['shape'],copy=True)
        output_shape1[0]=batch
        interpreter.resize_tensor_input(input_tensor_index1,input_shape1)
        interpreter.resize_tensor_input(input_tensor_index2,input_shape2)
        interpreter.resize_tensor_input(output_tensor_index,output_shape1)
        interpreter.allocate_tensors()
        
        input_scale1, input_zero_point1 = interpreter.get_input_details()[0]['quantization']
        input_scale2, input_zero_point2 = interpreter.get_input_details()[1]['quantization']
        data1=np.uint8(data1 / input_scale1 + input_zero_point1)
        data2=np.uint8(data2 / input_scale2 + input_zero_point2)
        try:
            interpreter.set_tensor(input_tensor_index1, data1)
            interpreter.set_tensor(input_tensor_index2, data2)
        except:
            interpreter.set_tensor(input_tensor_index1, data2)
            interpreter.set_tensor(input_tensor_index2, data1)
        start=time.perf_counter()
        interpreter.invoke()
        end=time.perf_counter()
        output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
        output_scale, output_zero_point = interpreter.get_output_details()[0]['quantization']
        output=output()
        output=np.round(output_scale * (output - output_zero_point))
        # index1=(output>59)
        # output[index1]=np.ceil(output[index1])
        # output=np.round(output)
        # index2=np.isin(output,[119.0,116.0,125.0,120.0])
        # output[index2]=output[index2]+1
        return output,end-start

# class tfmodel:
#     def __init__(self, model):
#         self.model=model

#     def quantize(self,prt=False):
#         kkk=256
#         interval=1
#         class data:
#             def __init__(self, shape):
#                 self.shape = shape
#             def data_set(self):
#                 iter2=0
#                 for _ in range(kkk*int((1/interval))):
#                     iter1=0;
#                     for _ in range(kkk*int((1/interval))):
#                         data=[]
#                         for s_index in range(len(self.shape)):
#                             s=self.shape[s_index]
#                             if s_index==0:
#                                 data.append((np.ones(s)*iter1).astype(np.float32))
#                             if s_index==1:
#                                 data.append((np.ones(s)*iter2).astype(np.float32))
#                         iter1=iter1+interval
#                   #data.append(np.random.randint(256, size=1).astype(np.float32))
#                         yield data
#                     iter2=iter2+interval
#         input_shape=[list(self.model.input.shape)]
#         input_shape[0][0]=1
#         datas=data(input_shape)
#         if prt:
#             print(list(datas.data_set))
#         #print(list(datas.data_set()))
#         converter1 = tf.lite.TFLiteConverter.from_keras_model(self.model)
#         converter1.optimizations = [tf.lite.Optimize.DEFAULT]
#         converter1.representative_dataset =datas.data_set
#         converter1.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#         converter1.target_spec.supported_types = [tf.int8]
#         converter1.inference_input_type = tf.uint8
#         converter1.inference_output_type = tf.uint8
#         tflite_model1 = converter1.convert()
#         self.tflite_model=tflite_model1
#         with open('model_quant.tflite', 'wb') as f:
#             f.write(tflite_model1)
#         os.system("edgetpu_compiler model_quant.tflite")
#         return
#     def evaluate_tflite_model(self,data,batch):
#         # Initialize TFLite interpreter using the model.
#         #import pdb;pdb.set_trace()
#         interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
#         interpreter.allocate_tensors()
        
#         input_tensor_index = interpreter.get_input_details()[0]["index"]
#         output_tensor_index = interpreter.get_output_details()[0]["index"]
#         shape1=np.array(interpreter.get_input_details()[0]['shape'],copy=True)
#         shape1[0]=batch
#         shape2=np.array(interpreter.get_output_details()[0]['shape'],copy=True)
#         shape2[0]=batch
#         interpreter.resize_tensor_input(input_tensor_index,shape1)
#         interpreter.resize_tensor_input(output_tensor_index,shape2)
#         interpreter.allocate_tensors()
        
#         scale, zero_point = interpreter.get_input_details()[0]['quantization']
#         data=np.uint8(data / scale + zero_point)
#         #data=np.uint8(data)
#         interpreter.set_tensor(input_tensor_index, data) 
        
#         output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
#         interpreter.invoke()
#         scale, zero_point = interpreter.get_output_details()[0]['quantization']
#         output=output()
#         output=scale * (output - zero_point)
#         index1=(output>59)
#         output[index1]=np.ceil(output[index1])
#         output=np.round(output)
#         index2=np.isin(output,[119.0,116.0,125.0,120.0])
#         output[index2]=output[index2]+1
#         return output

#     def evaluate_edgetpu_tflite_model(self,edgetpu_path,data,batch):
#         # Initialize TFLite interpreter using the model.
#         start1=time.perf_counter()
#         model_file = os.path.join(edgetpu_path)
#         interpreter = edgetpu.make_interpreter(model_file)
#         interpreter.allocate_tensors()
        
#         input_tensor_index = interpreter.get_input_details()[0]["index"]
#         output_tensor_index = interpreter.get_output_details()[0]["index"]
#         shape1=np.array(interpreter.get_input_details()[0]['shape'],copy=True)
#         shape1[0]=batch
#         shape2=np.array(interpreter.get_output_details()[0]['shape'],copy=True)
#         shape2[0]=batch
#         interpreter.resize_tensor_input(input_tensor_index,shape1)
#         interpreter.resize_tensor_input(output_tensor_index,shape2)
#         interpreter.allocate_tensors()
        
#         scale, zero_point = interpreter.get_input_details()[0]['quantization']
#         data=np.uint8(data / scale + zero_point)
#         interpreter.set_tensor(input_tensor_index, data)
        
#         output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
#         start2=time.perf_counter()
#         interpreter.invoke()
#         end2=time.perf_counter()
#         scale, zero_point = interpreter.get_output_details()[0]['quantization']
#         output=output()
#         output=scale * (output - zero_point)
#         index1=(output>59)
#         output[index1]=np.ceil(output[index1])
#         output=np.round(output)
#         index2=np.isin(output,[119.0,116.0,125.0,120.0])
#         output[index2]=output[index2]+1
#         end1=time.perf_counter()
#         return output