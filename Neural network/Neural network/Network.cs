namespace Neural_network
{
    internal class Network
    {
        public List<List<Neuron>> neurons;
        public Network(int[] sizes)
        {
            neurons = new List<List<Neuron>>();
            for (int i = 0; i < sizes.Length; i++)
            {
                List<Neuron> newlayer = new List<Neuron>();
                for (int k = 0; k < sizes[i]; k++)
                {
                    Neuron n;
                    if (i > 0)
                    {
                        double[] _weights = new double[neurons[i - 1].Count + 1];
                        Random r = new Random();
                        for (int j = 1; j < _weights.Length; j++)
                        {
                            _weights[j] = r.NextDouble();
                        }
                        n = new Neuron(neurons[i - 1], _weights, Activation_Functions.Id, Activation_Functions.Id_derivative, Activation_Functions.Id_inverse);
                    }
                    else
                        n = new Neuron(new List<Neuron>(), new double[0], Activation_Functions.Id, Activation_Functions.Id_derivative, Activation_Functions.Id_inverse);
                    newlayer.Add(n);
                }
                neurons.Add(newlayer);
            }
        }
        public double[] Propagate(double[] entries)
        {
            double[] returnvalue = new double[neurons[neurons.Count - 1].Count];
            for (int i = 0; i < entries.Length; i++)
                neurons[0][i].signal = entries[i];

            for (int i = 1; i < neurons.Count; i++)
                for (int j = 0; j < neurons[i].Count; j++)
                {
                    neurons[i][j].Forward_propagate();
                    if (i == neurons.Count - 1)
                        returnvalue[j] = neurons[i][j].signal;
                }

            return returnvalue;
        }

        public void Reset_back()
        {
            for (int i = neurons.Count - 1; i >= 0; i--)
                for (int j = 0; j < neurons[i].Count; j++)
                    neurons[i][j].back_signal = 0;
        }
        public double Back_propagate(double[] expected_output)
        {
            double s = 0;
            for (int i = neurons.Count - 1; i > 0; i--)
                for (int j = 0; j < neurons[i].Count; j++)
                {
                    if (i == neurons.Count - 1)
                    {
                        s += Math.Pow((neurons[i][j].signal - expected_output[j]), 2);
                        neurons[i][j].back_signal = 2 * (neurons[i][j].signal - expected_output[j]);
                    }
                    neurons[i][j].Back_propagate();
                }

            Reset_back();
            return (s);
        }
    }
}
