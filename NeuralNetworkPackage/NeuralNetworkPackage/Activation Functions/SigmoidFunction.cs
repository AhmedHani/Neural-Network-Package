using System;
using System.Collections.Generic;

namespace NeuralNetworkPackage
{
	public class SigmoidFunction : MathFunction
	{
		public SigmoidFunction ()
		{
		}

        /// <summary>
        /// Compute the sigmoid of the given input.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>double</returns>
		public double function (double input)
		{
			double sigmoid = 1 / (1 + Math.Exp (-input));
			return sigmoid;
		}

        /// <summary>
        /// Compute the derivative of the Sigmoid value.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>double</returns>
		public double derivative (double input)
		{
			double sigmoid = this.function (input);
			return sigmoid * (1 - sigmoid);
            
		}


	}
}

