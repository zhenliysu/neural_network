# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 08:30:22 2017

@author: li
"""

import mnist_loader

import network

if __name__=='__main__':
    print __name__
    for i in xrange(1,2):
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
        net = network.Network([784,30,10])
    

        net.SGD(training_data,30, 10, 1.0, test_data=test_data)
