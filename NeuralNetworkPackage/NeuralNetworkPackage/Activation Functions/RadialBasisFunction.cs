using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPackage.Activation_Functions
{
    public interface RadialBasisFunction
    {
        double function(List<double> values, List<double> centroids, double variance);
    }
}
