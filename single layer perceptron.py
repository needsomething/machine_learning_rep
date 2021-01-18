import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class single_layer_perceptron:
    data = np.zeros(1)
    hasil = np.zeros(1)
    weight = np.zeros(1)
    bias = np.zeros(1)
    waktu_pelatihan = 0
    learning_rate = 0.0
    y_pred = np.zeros(1)
    def __init__(self,data, hasil, weight, bias, waktu_pelatihan, learning_rate=1.0):
        self.data = data
        self.hasil = hasil
        self.weight = weight
        self.bias = bias
        self.waktu_pelatihan = waktu_pelatihan
        self.learning_rate = learning_rate
    
    def training(self,break_halfway=True):
        delta_weight = np.zeros(1)
        delta_bias = np.zeros(1)
        err = np.zeros(1)
        for i in range(self.waktu_pelatihan):
            self.y_pred = np.dot(self.data, self.weight) + self.bias

            self.y_pred[self.y_pred > 0] = 1
            self.y_pred[self.y_pred < 0] = 0

            err = self.hasil - self.y_pred

            if(np.sum(err) == 0 and break_halfway):
                break
            
            delta_weight = self.learning_rate * np.dot(np.transpose(self.data), err)
            delta_bias = self.learning_rate * np.sum(err)
            
            self.weight = self.weight + delta_weight
            self.bias = self.bias + delta_bias

            print("iterasi ke-", str(i) , err, self.weight, self.bias)
    
    def predict(self, predict_item):
        hasil = np.zeros(1)
        return_value = []
        for predict_item_target in predict_item:
            hasil = np.dot(predict_item_target, self.weight) + self.bias
            hasil = 1 if hasil > 0 else 0
            return_value.append(hasil)
            #print(str(predict_item_target[0]) + "AND" + str(predict_item_target[1]) + " = " + str(hasil))
        return return_value

def main():
    num_feature = 2 #banyak input yang diberikan untuk dipelajari
    num_enumerate = 100 #banyak perulangan sampai data sempurna

    x = np.array(([0,0], [0,1], [1,0], [1,1])) #input data
    y = np.array((0,0,0,1)) #hasil yang diinginkan

    w = np.zeros(num_feature) #weight (berat)
    b = np.zeros(1) #bias

    logic_AND = single_layer_perceptron(x, y, w, b, num_enumerate, learning_rate=0.5)
    logic_AND.training()

    y = np.array([0,1,1,1])

    logic_OR = single_layer_perceptron(x, y, w, b, num_enumerate, learning_rate=0.5)
    logic_OR.training()

    y = np.array([0,1,1,0])

    logic_XOR = single_layer_perceptron(x, y, w, b, num_enumerate, learning_rate=0.5)
    logic_XOR.training()

    x_test = [[0,0], [0,1], [1,0], [1,1]]

    temp =  logic_AND.predict(x_test)

    indeks = 0

    for x in x_test:
        x[0] = temp[indeks]
        indeks += 1

    temp = logic_OR.predict(x_test)

    indeks = 0

    for x in x_test:
        x[0] = temp[indeks]
        indeks += 1

    temp = logic_XOR.predict(x_test)

    print(temp)


if(__name__ == "__main__"):
    main()