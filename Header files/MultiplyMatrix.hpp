//this class is a function disguised as a class. this is because it might help later when doing GPGPU
#ifndef _MULTIPLY_MATRIX_HPP_
#define _MULTIPLY_MATRIX_HPP_

#include <iostream>
#include <vector>
#include <assert.h>

#include "Matrix.hpp"

using namespace std;

namespace utils
{
	class MultiplyMatrix
	{
	public:
		MultiplyMatrix(Matrix *a, Matrix *b);

		Matrix *execute();
	private:
		Matrix *a;
		Matrix *b;
		Matrix *c;
	};
}   


#endif

