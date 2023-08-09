#include "../NeuralNetwork/Neuron.hpp"

// Set Neuron input val
void Neuron::setVal(double val) {
	this->val = val;
	activate();
	derive();
}
//Constructor
Neuron::Neuron(double val){
	this->val = val;
	activate();
	derive();
}

/*	fast sigmoid functon
		f(x) = x / (1 + |x|)	*/
void Neuron::activate(){
	this->activatedVal = this->val / (1 + abs(this->val));
}

/*	derivitive for fast sigmoid function
		f'(x) = f(x) * (1 - f(x))	*/
void Neuron::derive(){
	this->derivedVal = this->activatedVal * (1 - (this->activatedVal));
}
