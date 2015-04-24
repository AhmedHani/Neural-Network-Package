using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPackage.Activation_Functions
{
    public class GaussianRadialBasisFunction : RadialBasisFunction
    {
        public GaussianRadialBasisFunction()
        {
        }

        public double function(List<double> values, List<double> centroids, double variance)
        {
            double standardDeviation = Math.Sqrt(variance);

            double sum = 0.0;

            for (int i = 0; i < values.Count; i++)
            {
                sum += (values[i] - centroids[i]) * (values[i] - centroids[i]);
            }

            return Math.Exp(-sum / 2 * variance);
        }
    }
}
