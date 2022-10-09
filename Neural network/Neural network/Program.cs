using Neural_network;
int entry = 3, outputs = 4;


Console.WriteLine("Testing Input ");
int m = 50000000;
int iterations = 1000;
Random r = new Random();
double[,] weights = new double[outputs, entry + 1];

for (int i = 0; i < outputs; i++)
    for (int j = 0; j <= entry; j++)
        weights[i, j] = r.NextDouble();
double avg = 0; 
for (int u = 0; u < iterations; u++)
{
    Network n = new Network(new int[] { entry, outputs }) ;
    for (int i = 0; i < m; i++)
    {
        double[] input = new double[entry], output = new double[outputs];
        for (int j = 0; j < entry; j++)
            input[j] = r.NextDouble();
        for (int j = 0; j < outputs; j++)
        {
            double s = weights[j, 0];
            for (int k = 0; k < entry; k++)
            {
                s += input[k] * weights[j, k + 1];
            }
            output[j] = s;
        }
        n.Propagate(input);
        double error = n.Back_propagate(output);
        /*if (error < 1e-5)
            for (int t = 0; t < outputs; t++)
            {
                for (int j = 0; j <= entry; j++)
                    Console.Write(Math.Round(weights[t, j], 4) + "/" + Math.Round(n.neurons.Last()[t].weights[j], 4) + "  ");
                Console.WriteLine();
            }*/
        if (error < 1e-7)
        {
            avg += i / (double)iterations;
            break;
        }
    }
}
Console.WriteLine(avg);
