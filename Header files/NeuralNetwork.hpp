#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

#include <iostream>
#include <vector>
#include <algorithm>
#include "MultiplyMatrix.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"


using namespace std;

class NeuralNetwork
{
public:

	NeuralNetwork(vector<int> topology);	// pass a topology to the neural network
	void setCurrentInput(vector<double> input); 
	void setCurrentTarget(vector<double> target) { this->target = target; }; 
	void feedForward(); 
	void backPropagation();
	void printToConsole();
	void setErrors();

	Matrix *getNeuronMatrix(int index) {return this->layers.at(index)->matrixifyVals();}; //return the layer at index and return unchanged values
	Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); }; //return the layer at index and return activated values
	Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); };//return the layer at index and return derived values
	Matrix *getWeightMatrix(int index) { return this->weightMatrices.at(index); };

	void setNeuronValue(int indexLayer, int indexNeuron, double val) { this->layers.at(indexLayer)->setVal(indexNeuron, val); } // sets the neuron values at the next hidden layer after the current layer

	double getTotalError() { return this->error; };
	vector<double> getErrors() { return this->errors; };
private:
	int					topologySize; //to store the topology
	vector<int>			topology;
	vector<Layer *>		layers;
	vector<Matrix *>	weightMatrices;
	vector<Matrix *>	gradientMatrices;
	vector<double>		input;
	vector<double>		target;
	double				error;
	vector<double>		errors;
	vector<double>		historicalErrors;
};
#endif
