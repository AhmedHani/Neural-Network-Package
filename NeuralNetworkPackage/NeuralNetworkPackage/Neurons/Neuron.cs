using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPackage.Neurons
{
    public abstract class Neuron
    {
        public abstract double computeOutput(List<double> input);
    }
}
