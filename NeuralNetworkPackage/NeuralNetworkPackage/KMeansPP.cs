using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPackage
{
    public class KMeansPP
    {
        private int numOfClusters;
        private List<List<double>> centroids;
        private List<List<double>> trainingSamples;

        public KMeansPP(int numOfClusters, List<List<double>> trainingSamples)
        {
            this.numOfClusters = numOfClusters;
            this.trainingSamples = trainingSamples;

            this.initializeCentroids();
        }

        public List<List<double>> run()
        {
            List<double> currentSeeds = new List<double>();
            List<double> oldSeeds = new List<double>();

            for (int i = 0; i < this.trainingSamples.Count; i++)
            {
                currentSeeds.Add(double.MaxValue);
                oldSeeds.Add(double.MinValue);
            }

            do
            {
                for (int i = 0; i < currentSeeds.Count; i++)
                {
                    oldSeeds.Add(currentSeeds[i]);
                }

                for (int i = 0; i < currentSeeds.Count; i++)
                {
                    currentSeeds[i] = this.getMinimumDistanceIndex(i);
                }

                for (int i = 0; i < this.centroids.Count; i++)
                {
                    this.centroids[i] = this.getMean(currentSeeds, i);
                }

            } while (!this.checkSimilarity(currentSeeds, oldSeeds));

            return this.centroids;
        }

        private void initializeCentroids()
        {
            Random rnd = new Random();
            this.centroids = new List<List<double>>();
            int initialCentroidIndex = (int)rnd.NextDouble() * this.trainingSamples.Count;

            this.centroids.Add(this.trainingSamples[initialCentroidIndex]);

            for (int cluster = 1; cluster < this.numOfClusters; cluster++)
            {
                List<double> probability = this.getCummulativeProbability();
                int index = this.getRandomIndex(probability, rnd.NextDouble());

                this.centroids.Add(trainingSamples[index]);
            }
        }

        private int getRandomIndex(List<double> probabilities, double probability)
        {
            for (int i = 0; i < probabilities.Count; i++)
            {
                if (probability < probabilities[i])
                {
                    return i;
                }
            }

            return probabilities.Count - 1;
        }

        private List<double> getCummulativeProbability()
        {
            List<double> cummulativeProbabilities = new List<double>();
            double error = this.getError();

            for (int i = 0; i < this.trainingSamples.Count; i++)
            {
                double minimumDistance = this.getMinimumDistance(i);
                cummulativeProbabilities.Add(minimumDistance / error);
            }

            for (int i = 1; i < this.trainingSamples.Count; i++)
            {
                cummulativeProbabilities[i] += cummulativeProbabilities[i - 1];
            }

            return cummulativeProbabilities;

        }

        private double getMinimumDistance(int trainingSamplesIndex)
        {
            double minimum = double.MaxValue;

            for (int i = 0; i < centroids.Count; i++)
            {
                double distance = this.getEuclideanDistance(this.centroids[i], this.trainingSamples[trainingSamplesIndex]);
                minimum = Math.Min(minimum, distance);
            }

            return minimum;
        }

        private int getMinimumDistanceIndex(int trainingSamplesIndex)
        {
            double minimum = double.MaxValue;
            int minimumValueIndex = -1;

            for (int i = 0; i < this.centroids.Count; i++)
            {
                double currentDistance = this.getEuclideanDistance(this.centroids[i], this.trainingSamples[trainingSamplesIndex]);

                if (currentDistance < minimum)
                {
                    minimumValueIndex = i;
                    minimum = currentDistance;
                }
            }

            return minimumValueIndex;
        }

        private double getEuclideanDistance(List<double> first, List<double> second)
        {
            double sum = 0.0;

            for (int i = 0; i < first.Count; i++)
            {
                sum += Math.Pow(first[i] - second[i], 2);
            }

            return Math.Sqrt(sum);
        }

        private double getError()
        {
            double error = 0.0;

            for (int i = 0; i < this.trainingSamples.Count; i++)
            {
                error += this.getMinimumDistance(i);
            }

            return error;
        }

        private bool checkSimilarity(List<double> first, List<double> second)
        {
            for (int i = 0; i < first.Count; i++)
            {
                if (first[i] != second[i])
                {
                    return false;
                }
            }

            return true;
        }

        private List<double> getMean(List<double> seeds, int index)
        {
            int size = 0;
            List<double> currentMean = new List<double>();

            for (int i = 0; i < this.centroids.Count; i++)
            {
                currentMean.Add(0.0);
            }

            for (int i = 0; i < this.trainingSamples.Count; i++)
            {
                if (seeds[i] == index)
                {
                    size++;

                    for (int j = 0; j < currentMean.Count; j++)
                    {
                        currentMean[j] += this.trainingSamples[i][j];
                    }
                }
            }

            for (int i = 0; i < currentMean.Count; i++)
            {
                currentMean[i] = currentMean[i] / size;
            }

            return currentMean;
        }

        public List<double> getStandardDeviation()
        {
            List<double> widths = new List<double>();

            double sum = 0.0;
            int size = 0;

            for (int i = 0; i < this.centroids.Count - 1; i++)
            {
                for (int j = i + 1; j < this.centroids.Count; j++)
                {
                    double distance = this.getEuclideanDistance(centroids[i], centroids[j]);
                    sum += distance;
                    size++;
                }
            }

            double average = sum / size;
            double width = average;

            for (int i = 0; i < this.centroids.Count; i++)
            {
                widths[i] = width;
            }

            return widths;
        }
    }
}
