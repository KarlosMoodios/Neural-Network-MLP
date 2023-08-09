#ifndef _NEURON_HPP_
#define _NEURON_HPP_
#include <iostream>
#include <math.h>
using namespace std;

class Neuron
{
public:
	Neuron(double val);
	void setVal(double v);
	void activate();	/*	fast sigmoid functon
							f(x) = x / (1 + |x|)	*/ 
	void derive();		/*	derivitive for fast sigmoid function
							f'(x) = f(x) * (1 - f(x))	*/
	//	get functions
	double getVal() {return this->val;}
	double getActivatedVal() {return this->activatedVal;}
	double getDerivedVal() {return this->derivedVal;}

private:
	double val;				//	1.5
	double activatedVal;	//	0< = x = >1
	double derivedVal;		//	derivative of activatedVal
};
#endif
