using NeuralNetworkPackage.Activation_Functions;
using NeuralNetworkPackage.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPackage.Neural_Network
{
    public class RadialBasisHiddenNeuron : Neuron
    {
        private List<double> input;
        private List<double> centroids;
        private double width;
        private double output;
        private RadialBasisFunction activationFunction;

        public List<double> Centroids
        {
            get { return this.centroids; }
        }

        public double Width
        {
            get { return this.width; }
        }

        public RadialBasisFunction ActivationFunction
        {
            get { return this.activationFunction; }
        }

        public RadialBasisHiddenNeuron(int numOfInput, RadialBasisFunction activationFunction)
        {
            this.activationFunction = activationFunction;
            this.init(numOfInput);

            
        }

        public RadialBasisHiddenNeuron(List<double> centroids, RadialBasisFunction activationFunction)
        {
            this.activationFunction = activationFunction;
            this.init(centroids.Count);

            this.centroids = centroids;
        }

        public RadialBasisHiddenNeuron(List<double> centroids, double width, RadialBasisFunction activationFunction)
        {
            this.activationFunction = activationFunction;
            this.init(centroids.Count);

            this.centroids = centroids;
            this.width = width;
        }

        private void init(int numOfInput)
        {
            this.centroids = new List<double>();
            
            for (int i = 0; i < numOfInput; i++)
            {
                this.input.Add(0);
                this.centroids.Add(0);
            }

            this.width = 0.0;
            this.output = 0.0;
        }

        public void update(List<double> centroids, double width)
        {
            this.centroids = centroids;
            this.width = width;
        }

        public override double computeOutput(List<double> input)
        {
            return this.activationFunction.function(input, this.centroids, this.width);
        }
    }
}
