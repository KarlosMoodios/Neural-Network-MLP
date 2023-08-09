#include "MultiplyMatrix.hpp"

utils::MultiplyMatrix::MultiplyMatrix(Matrix *a, Matrix *b) {
	this->a = a;
	this->b = b;

	if (a->getNumCols() != b->getNumRows()) {//error check 
		cerr << "A_Cols: " << a->getNumRows() << "!= B_Rows: " << b->getNumCols() << endl;
		assert(false);
	}
	//create new matrix called c, take the number of rows from matrix 'a' and the cols from matrix 'b'
	this->c = new Matrix(a->getNumRows(), b->getNumCols(), false);
}

Matrix *utils::MultiplyMatrix::execute() {
	// known as an ijk loop for multiplying matrices
	for (int i = 0; i < a->getNumRows(); i++){ //get number of ROWS from matrix a
		for (int j = 0; j < b->getNumCols(); j++) {//get number of COLS from matrix b
			for (int k = 0; k < b->getNumRows(); k++) {//get number of ROWS from matrix b
				double p = this->a->getValue(i, k) * this->b->getValue(k, j); //multiply matrix a value (i, k) by matrix b value (k, j)
				double newVal = this->c->getValue(i, j) + p; // get matrix c value (i, j) and add them the 'p'
				this->c->setValue(i, j, newVal); //save the new value in matrix c (i, j)
			}
		}
	}
	return this->c;
}