using NeuralNetworkPackage.Activation_Functions;
using NeuralNetworkPackage.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPackage.Neural_Network
{
    public class RadialBasisNeuralNetwork
    {
        private const int INPUT_LAYER = 0;
        private const int HIDDEN_LAYER = 1;
        private const int OUTPUT_LAYER = 2;
        private const int NUM_OF_LAYERS = 3;

        private List<int> numOfNeuronsPerLayer;
        private List<List<Neuron>> network;
        private bool hiddenNeuronsTraining;

        public RadialBasisNeuralNetwork(List<int> numOfNeuronsPerLayer)
        {
            this.numOfNeuronsPerLayer = numOfNeuronsPerLayer;
            this.network = new List<List<Neuron>>();
        }

        public void initializeHiddenLayer(RadialBasisFunction radialBasisFunction) 
        {
            for (int i = 0; i < numOfNeuronsPerLayer[HIDDEN_LAYER]; i++) 
            {
                this.network[HIDDEN_LAYER].Add(new RadialBasisHiddenNeuron(numOfNeuronsPerLayer[HIDDEN_LAYER], radialBasisFunction));
            }
        }

        public void initializeHiddenLayer(List<List<double>> centroids, RadialBasisFunction radialBasisFunction)
        {
            for (int i = 0; i < centroids.Count; i++)
            {
                this.network[HIDDEN_LAYER].Add(new RadialBasisHiddenNeuron(centroids[i], radialBasisFunction));
            }
        }

        public void initializeHiddenLayer(List<List<double>> centroids, double width, RadialBasisFunction radialBasisFunction)
        {
            for (int i = 0; i < centroids.Count; i++)
            {
                this.network[HIDDEN_LAYER].Add(new RadialBasisHiddenNeuron(centroids[i], width, radialBasisFunction));
            }
        }

        public void initializeOutputLayer(MathFunction activationFunction)
        {
            for (int i = 0; i < numOfNeuronsPerLayer[OUTPUT_LAYER]; i++) 
            {
                this.network[OUTPUT_LAYER].Add(new FeedfowardNeuron(numOfNeuronsPerLayer[INPUT_LAYER], activationFunction));
            }
        }

        public void initializeOutputLayer(List<double> weights, MathFunction activationFunction)
        {
            for (int i = 0; i < numOfNeuronsPerLayer[OUTPUT_LAYER]; i++)
            {
                this.network[OUTPUT_LAYER].Add(new FeedfowardNeuron(weights, activationFunction));
            }
        }

        public void initializeOutputLayer(List<double> weights, double bias, MathFunction activationFunction)
        {
            for (int i = 0; i < numOfNeuronsPerLayer[OUTPUT_LAYER]; i++)
            {
                this.network[OUTPUT_LAYER].Add(new FeedfowardNeuron(weights, bias, activationFunction));
            }
        }

        public List<double> computeOutput(List<double> input)
        {
            List<double> output = new List<double>();
            List<double> hiddenLayerOutput = new List<double>();

            for (int layerIndex = 1; layerIndex < NUM_OF_LAYERS; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < numOfNeuronsPerLayer[layerIndex]; neuronIndex++)
                {
                    if (layerIndex == 1)
                    {
                        RadialBasisHiddenNeuron radialBasis = (RadialBasisHiddenNeuron)this.network[layerIndex][neuronIndex];
                        double radialBasisOutput = radialBasis.computeOutput(input);
                        hiddenLayerOutput.Add(radialBasisOutput);
                    }

                    else
                    {
                        FeedfowardNeuron feedforwardNeuron = (FeedfowardNeuron)this.network[layerIndex][neuronIndex];
                        double feedforwardOutput = feedforwardNeuron.computeOutput(hiddenLayerOutput);
                        output.Add(feedforwardOutput);
                    }
                }
            }

            return output;
        }

        public void train(List<List<double>> trainingSamples, List<List<double>> trainingLabels, double learningRate, LearningAlgorithm learningAlgorithm)
        {
            if (hiddenNeuronsTraining == false)
            {
                KMeansPP kmeansPlusPlus = new KMeansPP(numOfNeuronsPerLayer[HIDDEN_LAYER], trainingSamples);
                List<List<double>> centroids = kmeansPlusPlus.run();
                List<double> widths = kmeansPlusPlus.getStandardDeviation();
                RadialBasisHiddenNeuron radialBasis;

                for (int j = 0; j < numOfNeuronsPerLayer[HIDDEN_LAYER]; j++)
                {
                    radialBasis = (RadialBasisHiddenNeuron)this.network[HIDDEN_LAYER][j];
                    radialBasis.update(centroids[j], widths[j]);

                    this.network[HIDDEN_LAYER][j] = radialBasis;
                }

                hiddenNeuronsTraining = true;
            }

            for (int i = 0; i < trainingSamples.Count; ++i)
            {
                List<double> output = this.computeOutput(trainingSamples[i]);
                //this.network = learningAlgorithm.learn(learningRate, trainingSamples[i], trainingLabels[i], this.network);
            }
        }

      
    }
}
