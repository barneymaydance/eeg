from __future__ import division
import numpy
import math
from scipy import spatial


class RSM:
    def __init__(
            self,
            input,
            num_visible,
            num_hidden,
            learning_rate=0.1
    ):
        # Initial state of each variable
        self.input = input
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        # Create Random generator
        self.numpy_rng = numpy.random.RandomState()

        # Initial Weight which is uniformely sampled from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
        # self.weights = numpy.asarray(
        #     self.numpy_rng.uniform(
        #         low=-4 * numpy.sqrt(6. / (self.num_visible + self.num_hidden)),
        #         high=4 * numpy.sqrt(6. / (self.num_visible + self.num_hidden)),
        #         size=(self.num_visible, self.num_hidden)
        #     )
        # )
        mu, sigma = 0, numpy.sqrt(0.01)
        self.weights = self.numpy_rng.normal(mu, sigma, (
            self.num_visible, self.num_hidden))
        # self.weights=0.1*numpy.random.randn(self.num_visible,self.num_hidden)
        # Inital hidden Bias
        self.hbias = numpy.zeros(self.num_hidden)

        # Inital visible Bias
        self.vbias = numpy.zeros(self.num_visible)

        self.delta_weights = numpy.zeros((self.num_visible, self.num_hidden))
        self.delta_hbias = numpy.zeros(self.num_hidden)
        self.delta_vbias = numpy.zeros(self.num_visible)

    # sigmoid function:
    def sigmoid(self, x):
        return 1. / (1 + numpy.exp(-x))

    # softmax function:
    # def softmax(self, x):
    #     e_x = numpy.exp(x)
    #     return e_x / numpy.sum(e_x, axis=1)[:,None]
    def softmax(self, x):
        numerator = numpy.exp(x)
        denominator = numerator.sum(axis=1)
        denominator = denominator.reshape((x.shape[0], 1))
        softmax = numerator / denominator
        return softmax

    # Calculate and return Positive hidden states and probs
    def positiveProb(self, visible, D):
        pos_hidden_activations = numpy.dot(visible, self.weights) + numpy.outer(D, self.hbias)
        pos_hidden_probs = self.sigmoid(pos_hidden_activations)
        pos_hidden_states = self.numpy_rng.binomial(
            n=1,
            p=pos_hidden_probs,
            size=pos_hidden_probs.shape,
        )
        return [pos_hidden_states, pos_hidden_probs]

    # Calculate and return Negative hidden states and probs
    def negativeProb(self, data, hidden, D, k=1):
        for i in range(k):
            v1_activations = numpy.dot(hidden, self.weights.T) + self.vbias
            v1_probs = self.softmax(v1_activations)
            v1_sample= numpy.zeros(v1_probs.shape)
            for i_d, d in enumerate(D):
                v1_sample[i_d] = self.numpy_rng.multinomial(
                    d,
                    v1_probs[i_d]
                )
            # Get back to calculate hidden again
            hidden, hidden_probs = self.positiveProb(v1_sample, numpy.sum(v1_sample, axis=1))
        return [v1_probs, v1_sample, hidden_probs, hidden]

    # Train RMB model
    def train(self, max_epochs=15, batch_size=10, step=1, weight_cost=0.0002, momentum=0.9):
        data=self.input
        for epoch in range(max_epochs):
            # Divide in to minibatch
            total_batch = int(math.ceil(data.shape[0] / batch_size))
            reconstruction_error=0
            # step=int(step*(1.0+9*epoch/(max_epochs-1)))
            print ("cd = {}".format(step))
            # Loop for each batch
            for batch_index in range(total_batch):
                # Get the data for each batch
                tmpData = data[batch_index * batch_size: (batch_index + 1) * batch_size]
                num_examples = tmpData.shape[0]
                # D
                D=numpy.sum(tmpData,axis=1)
                # Caculate positive probs and Expectation for Sigma(ViHj) data
                pos_hidden_states, pos_hidden_probs = self.positiveProb(tmpData,D)
                pos_associations = numpy.dot(tmpData.T, pos_hidden_probs)

                # Calculate negative probs and Expecatation for Sigma(ViHj) recon with k = 1,....
                neg_visible_probs,neg_visible_states, neg_hidden_probs, neg_hidden_states = self.negativeProb(tmpData, pos_hidden_states,D,k=step)
                neg_associations = numpy.dot(neg_visible_states.T, neg_hidden_probs)

                # Update weight
                self.delta_weights = momentum * self.delta_weights + self.learning_rate * (
                    (pos_associations - neg_associations) / num_examples - weight_cost * self.weights)
                self.delta_vbias = momentum * self.delta_vbias + (numpy.sum(tmpData, axis=0) - numpy.sum(neg_visible_states,axis=0)) * (self.learning_rate / num_examples)
                self.delta_hbias = momentum * self.delta_hbias + (numpy.sum(pos_hidden_probs, axis=0) - numpy.sum(neg_hidden_probs,axis=0)) * (self.learning_rate / num_examples)
                self.weights += self.delta_weights
                self.vbias += self.delta_vbias
                self.hbias += self.delta_hbias

                # self.weights += (self.learning_rate /num_examples * (pos_associations - neg_associations))
                # self.vbias += (self.learning_rate /num_examples * (numpy.sum(tmpData, axis=0) - numpy.sum(neg_visible_states,axis=0)))
                # self.hbias += (self.learning_rate /num_examples * (numpy.sum(pos_hidden_probs, axis=0) - numpy.sum(neg_hidden_probs,axis=0)))

                reconstruction_error += numpy.square(tmpData - neg_visible_states).sum()
            print('Epoch: {}, Error={}'.format(epoch, reconstruction_error))

    # # calculate distance between papers
    # def recommend(self, testcase, data, i, Rank=1):
    #     d=testcase.sum()
    #     D=data.sum(axis=1)
    #     testHidden = self.getHiddenPro(testcase, d)
    #     tmpHiddens = self.getHiddenPro(data,D)
    #     distance = []
    #     for tmpHidden in tmpHiddens:
    #     	distance.append(spatial.distance.euclidean(testHidden, tmpHidden))
    #     distance[i]=numpy.inf
    #     ind = numpy.argsort(distance)[:Rank]
    #     return ind

    def recommendByTraindata(self, testcase,data, Rank=1):
        d=testcase.sum()
        testHidden = self.getHiddenPro(testcase, d)
        D = data.sum(axis=1)
        tmpHiddens=self.getHiddenPro(data,D)
        distance = []
        for tmpHidden in tmpHiddens:
        	distance.append(spatial.distance.euclidean(testHidden, tmpHidden))
        ind = numpy.argsort(distance)[:Rank]
        return ind

    def cosineRecom(self, testcase, data, w, hb,Rank=5):
        d=testcase.sum()
        D=data.sum(axis=1)
        testHidden = self.getHiddenPro(testcase, w, hb, d)
        tmpHiddens = self.getHiddenPro(data, w, hb, D)
        distance = []
        for tmpHidden in tmpHiddens:
            distance.append(spatial.distance.cosine(testHidden, tmpHidden))
        ind = numpy.argsort(distance)[:Rank]
        return ind

    def getHiddenPro(self, visible, D):
        hidden_activations = numpy.dot(visible, self.weights) + numpy.outer(D,self.hbias)
        hidden_probs = self.sigmoid(hidden_activations)
        return hidden_probs

    def saveRsmWeights(self):
        filename = "newsgroups_" + str(self.num_visible) + "_" + str(self.num_hidden) + "_weights.bin"
        with open(filename, "wb") as file:
            numpy.savez(file=file,weights=self.weights, hbias=self.hbias,vbias=self.vbias)

    def saveTrainOutput(self,data ):
        output = self.getHiddenPro(data, numpy.sum(data,axis=1))
        filename="newsgroups_"+str(self.num_visible)+"_"+str(self.num_hidden)+"_output.bin"
        with open(filename, "wb") as file:
            numpy.savez(file=file, output=output)


