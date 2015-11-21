__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from PIL import Image
def load_data(config):
    import scipy.io as sio
    data_obj = sio.loadmat(config["train_file"])
    print "Loaded data..."
    return data_obj['price_move']


class HMM:

    def __init__(self,config):
        self.y = load_data(config)
        self.dsize = len(self.y)
        self.y = (1 + self.y)/2
        self.pr = config["prior"]
        self.p_t = config["p_trans"]
        self.p_e = config["p_emit"]
        self.file_to_save = config["file_to_save"]
        print "HMM object created and initialized"

    def plot(self,data):
        n = len(data)
        x = range(n)
        width = 1/1.5
        plt.bar(x,data,width)
        #plt.plot(data)
        plt.savefig(self.file_to_save)
        print "Plotted data"

    def upward_belief_prop(self):
        t,e,y,d = [self.p_t, self.p_e, self.y, self.dsize]
        m = []
        m.append((self.pr.T)*e[:,y[0]])
        for i in range(1,d):
            m.append((e[:,y[i]])*(t.dot(m[i-1])))
        self.p_y = sum(m[-1])
        return m

    def downward_belief_prop(self):
        t,e,y,d = [self.p_t, self.p_e, self.y, self.dsize]
        m = []
        m.append(np.array([1,1]))
        m.append(t.dot(e[:,y[-1]]))
        for i in range(2,d):
            m.append(t.dot(e[:,y[-i]]*m[i-1]))
        return m

    def compute_posterior(self,o_value):
        result = []
        self.f_m = self.upward_belief_prop()
        self.b_m = self.downward_belief_prop()
        for i in range(self.dsize):
            result.append(self.f_m[i][o_value]*self.b_m[-i-1][o_value])
        result = result/self.p_y
        print "Successfully ran Forward-Backward algorithm !"
        print "Probability that week 39 is good is :",result[-1]
        print result
        return result

    def run(self):
        result = self.compute_posterior(1)
        self.plot(result)

if __name__ == "__main__":
   config = {}
   q = 0.7
   config["train_file"]       = "/Users/chinna/Downloads/sp500.mat"
   config["p_trans"]          = np.array([[0.8, 0.2],[0.2, 0.8]])
   config["p_emit"]           = np.array([[q, 1-q],[1-q, q]])
   config["prior"]            = np.array([[0.8, 0.2]])
   config["file_to_save"]     = "fig1.jpeg"
   hmm = HMM(config)
   hmm.run()
