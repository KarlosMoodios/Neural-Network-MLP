// main.cpp
#include "../NeuralNetwork/Neuron.hpp"
#include "../NeuralNetwork/Matrix.hpp"
#include "../NeuralNetwork/NeuralNetwork.hpp"
#include "../NeuralNetwork/Layer.hpp"
#include <iostream>
#include <vector>
#include <conio.h>

using namespace std;
	
int main(int argc, char **argv) {
	// Set values of the neurons
	vector<double> input;
	input.push_back(1);
	input.push_back(0);// 3 input neurons
	input.push_back(1);

	// set the target for the output
	vector<double> target;
	target.push_back(1.00);	
	target.push_back(0.00); // target for the output neurons
	target.push_back(1.00);
	
	// Create the topology
	vector<int> topology; 
	topology.push_back(3); // input neurons
	topology.push_back(2); // hidden
	topology.push_back(4); // hidden
	topology.push_back(2); // hidden
	topology.push_back(3); // output neurons
	
	// Create new NeuralNetwork
	NeuralNetwork *nn = new NeuralNetwork(topology);

	// Set the input
	nn->setCurrentInput(input);
	// Set target output
	nn->setCurrentTarget(target);

	// Training the network
	for (int i = 0; i < 1000; i++) {
		// Perform a feedforward
		nn->feedForward();
		//Set the errors
		nn->setErrors();
		// Print the NeuralNetwork to the console window 
		//nn->printToConsole();
		// Print total error
		//cout << "Epoch: " << i << " Total Error: " << nn->getTotalError() << endl;
		// Back propagate the network to update the weights of each neuron for the next pass
		nn->backPropagation(); {
			
			//if (i == 0 || i == 999) {
				cout << "=====================================================================================================" << endl;
				//nn->printToConsole();
				cout << "Epoch: " << i << " Total Error: " << nn->getTotalError() << endl;
			//}
		}
	}
	
	return 0;
}