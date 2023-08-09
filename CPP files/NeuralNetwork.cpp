#include "../NeuralNetwork/NeuralNetwork.hpp"
#include <conio.h>

void NeuralNetwork::backPropagation() {
	vector<Matrix *> newWeights;
	Matrix *gradient;

	// calculating the gradients from the output to hidden layer
	int outputLayerIndex		= this->layers.size() - 1;
	Matrix *derivedValuesYToZ	= this->layers.at(outputLayerIndex)->matrixifyDerivedVals(); // 
	Matrix *gradientsYToZ = new Matrix(1, this->layers.at(outputLayerIndex)->getNeurons().size(), false);

		for (int i = 0; i < 3; i++) {
			double d = derivedValuesYToZ->getValue(0, i);
			double e = this->errors.at(i);
			double g = d * e; // multiply the derived vlaues of the output layer (d) by the errors of each neuron (e) and save it as the graident (g)
			gradientsYToZ->setValue(0, i, g);
		}

	int lastHiddenLayerIndex			= outputLayerIndex - 1;
	Layer *lastHiddenLayer				= this->layers.at(lastHiddenLayerIndex);
	Matrix *weightsOutputToHidden		= this->weightMatrices.at(outputLayerIndex - 1);
	Matrix *deltaOutputToHidden			= (new utils::MultiplyMatrix(gradientsYToZ->transpose(), lastHiddenLayer->matrixifyActivatedVals()))->execute()->transpose();
	Matrix *newWeightsOutputToHidden	= new Matrix(deltaOutputToHidden->getNumRows(), deltaOutputToHidden->getNumCols(), false);
	
	for (int r = 0; r < deltaOutputToHidden->getNumRows(); r++) {
		for (int c = 0; c < deltaOutputToHidden->getNumCols(); c++) {
			double originalWeight	= weightsOutputToHidden->getValue(r, c);
			double deltaWeight		= deltaOutputToHidden->getValue(r, c);
			newWeightsOutputToHidden->setValue(r, c, (originalWeight - deltaWeight));
			/*cout << "\nDelta output to hidden, R= " << r << ", C= " << c << endl;
			cout << "Original weight = " << originalWeight << ", Delta weight = " << deltaWeight << endl;*/
		}
	}
	newWeights.push_back(newWeightsOutputToHidden);
	//cout << "\nNew weights for the output to hidden layer: " << endl;
	//newWeightsOutputToHidden->printToConsole();
	

	gradient = new Matrix(gradientsYToZ->getNumRows(), gradientsYToZ->getNumCols(), false);

	for (int r = 0; r < gradientsYToZ->getNumRows(); r++) {
		for (int c = 0; c < gradientsYToZ->getNumCols(); c++) {
			gradient->setValue(r, c, gradientsYToZ->getValue(r, c));
			/*cout << "\ngradientsYToZ, R= " << r << ", C= " << c << endl;
			cout << "Gradient = " << gradient << endl;*/
		}
	}

	//cout << "\nOutput to hidden, New Weights" << endl;
	//newWeightsOutputToHidden->printToConsole();
	// end of output to hidden layer

	//cout << "\n============================END OF OUTPUT TO HIDDEN LAYER============================" << endl;



	//Looping from the last hidden layer down to the input layer
	for (int i = (outputLayerIndex - 1); i > 0; i--) {
		Layer *l					= this->layers.at(i);
		Matrix *activatedHidden		= l->matrixifyActivatedVals();
		Matrix *derivedHidden		= l->matrixifyDerivedVals();
		Matrix *derivedGradients	= new Matrix(1, l->getNeurons().size(), false);
		Matrix *weightMatrix		= this->weightMatrices.at(i);
		Matrix *originalWeight		= this->weightMatrices.at(i - 1);
		
		for (int r = 0; r < weightMatrix->getNumRows(); r++) {
			double sum = 0;
			for (int c = 0; c < weightMatrix->getNumCols(); c++) {
				double p = gradient->getValue(0, c) * weightMatrix->getValue(r, c);
				sum += p;
			}
			double g = sum * activatedHidden->getValue(0, r);
			derivedGradients->setValue(0, r, g);
		}

		Matrix *leftNeurons			= (i - 1) == 0 ? this->layers.at(0)->matrixifyVals() : this->layers.at(i - 1)->matrixifyActivatedVals();
		Matrix *deltaWeights		= (new utils::MultiplyMatrix(derivedGradients->transpose(), leftNeurons))->execute()->transpose();
		Matrix *newWeightsHidden	= new Matrix(deltaWeights->getNumRows(), deltaWeights->getNumCols(), false);
		
		for (int r = 0; r < newWeightsHidden->getNumRows(); r++) {
			for (int c = 0; c < newWeightsHidden->getNumCols(); c++) {
				double w = originalWeight->getValue(r, c);
				double d = deltaWeights->getValue(r, c);
				double n = w - d;
				newWeightsHidden->setValue(r, c, n);
				//cout << "\nNew Weights Hidden, R= " << r << ", C= " << c << endl;
				//cout << "New weight: " << n << endl;
			}
		}
		//cout << "\nNew Weights Hidden to input = " << endl;
		//newWeightsHidden->printToConsole();

		gradient = new Matrix(derivedGradients->getNumRows(), derivedGradients->getNumCols(), false);
		for (int r = 0; r < derivedGradients->getNumRows(); r++) {
			for (int c = 0; c < derivedGradients->getNumCols(); c++) {
				/*cout << "\nNumber of Rows in this matrix = " << derivedGradients->getNumRows() << endl;
				cout << "Number of Columns in this matrix = " << derivedGradients->getNumCols() << endl;*/
				gradient->setValue(r, c, derivedGradients->getValue(r, c));
				//cout << "Derived Gradients, R= " << r << ", C= " << c << " Gradient matrix: " << gradient << endl;
			}
		}
		//cout << "\nPrint the gradient matrix" << endl;
		//gradient->printToConsole();
		newWeights.push_back(newWeightsHidden);
	}
	//cout << "\nDone backProp =)" << endl;
	//cout << "\nNew Weight Size: " << newWeights.size() << endl;
	//cout << "Old Weight Size: " << this->weightMatrices.size() << endl;
	reverse(newWeights.begin(), newWeights.end());
	this->weightMatrices = newWeights;
}

void NeuralNetwork::setErrors() {
	if (this->target.size() == 0) {
		cerr << "No target for this neural network " << endl;
		assert(false);
	}

	if (this->target.size() != this->layers.at(this->layers.size() - 1)->getNeurons().size()) {
		cerr << "Target size is not the same as output layer size: " << this->layers.at(this->layers.size() - 1)->getNeurons().size() << endl;
		assert(false);
	}

	this->error = 0.00;
	int outputLayerIndex = this->layers.size() - 1;
	vector<Neuron *> outputNeurons = this->layers.at(outputLayerIndex)->getNeurons();
	for (int i = 0; i < target.size(); i++) {
		double tempErr = (outputNeurons.at(i)->getActivatedVal() - target.at(i));
		errors.push_back(tempErr);
		this->error += tempErr;
	}

	historicalErrors.push_back(this->error);
}

void NeuralNetwork::feedForward() {
	//loop through each layer from the input layer and get the values of the selected layer
	for (int i = 0; i < (this->layers.size() - 1); i++) {
		Matrix *a = this->getNeuronMatrix(i); // input layer. hidden layer(s) otherwise

		if (i != 0) {//check to see if we are on the input layer or on any other layer between the input and the last hidden layer
			a = this->getActivatedNeuronMatrix(i); //if we are not on the input layer get the activated values of the layer
		}

		Matrix *b = this->getWeightMatrix(i); //set matrix b to getWeightMatrix
		Matrix *c = (new utils::MultiplyMatrix(a, b))->execute(); //mutliply Matrices a and b

		
		for (int c_index = 0; c_index < c->getNumCols(); c_index++) {
			this->setNeuronValue(i + 1, c_index, c->getValue(0, c_index)); //set the result of c as the values in the first hidden layer
		}
	}

}

void NeuralNetwork::printToConsole() {
	for (int i = 0; i < this->layers.size(); i++) { //loop through layers
		cout << "LAYER: " << i << endl; // print which layer we're at
		if (i == 0) { // if i is equal to 0 is the input neurons
			Matrix *m = this->layers.at(i)->matrixifyVals(); // get the raw values from the input neurons
			m->printToConsole(); // print them to the console
		}
		else { 
			Matrix *m = this->layers.at(i)->matrixifyActivatedVals(); // otherwise get the activated values
			m->printToConsole(); //print them to the console
		}
		cout << "=============================" << endl; // seperater

		if (i < this->layers.size() - 1) {// loop to exclude the output layer
			cout << "Weight Matrix: " << i << endl; // print the layer index of the weight matrix 
			this->getWeightMatrix(i)->printToConsole(); // print the matrix tot he console
		}
		cout << "=============================" << endl;//seperater.
	}
}

void NeuralNetwork::setCurrentInput(vector<double> input) {
	this->input = input;

	for (int i = 0; i < input.size(); i++) {
		this->layers.at(0)->setVal(i, input.at(i)); // set the value of the neurons
	}
		
}

NeuralNetwork::NeuralNetwork(vector<int> topology) {
	this->topology = topology;
	this->topologySize	= topology.size();

	for (int i = 0; i < topologySize; i++) {
		Layer *l = new Layer(topology.at(i)); // tell the layer how many neurons to make
		this->layers.push_back(l);	// send the values of the new layer *l to the layers vector
	}
	for (int i = 0; i < topologySize - 1; i++) { // we dont want to include the output neurons.
		Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true); // rows, colums = input and hidden layer.  TRUE = random values for weights
		this->weightMatrices.push_back(m); // push the new matrix to the weightMatrices vector
	}
}