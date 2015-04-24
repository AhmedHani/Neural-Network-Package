using System;

namespace NeuralNetworkPackage
{
	public interface MathFunction
	{
		double function (double input);
		double derivative (double input);
	}
}

