using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_network
{
    internal class Neuron
    {
        public double signal=0;
        public double[] weights;
        double alpha=0.01;
        double alpha0 = 0.001;
        public Func<double, double> Activation;
        public Func<double, double> Activation_derivative;
        public Func<double, double> Activation_inverse;
        public double back_signal = 0;
        List<Neuron> parents;
        public Neuron(List<Neuron> _parents, double[] _weights,  Func<double,double> _activation, Func<double,double> _activation_derivative, Func<double,double> _activation_inverse, double _signal= 0)
        {
            parents = _parents;
            weights = _weights;
            signal = _signal;
            Activation = _activation;
            Activation_derivative = _activation_derivative;
            Activation_inverse = _activation_inverse;
        }

        public void Forward_propagate()
        {
            double sum = weights[0]; 
            for( int i =1; i< weights.Length; i++)
                sum += weights[i] * parents[i-1].signal;
            signal = Activation(sum);
        }
        
        public void Update_weights()
        {
            weights[0] -= back_signal*alpha0;
            for (int i = 1; i < weights.Length; i++)
            {
                weights[i] -= back_signal * alpha * parents[i - 1].signal;
            }
        }

        public void Back_propagate()
        {
            back_signal *= Activation_derivative(Activation_inverse(signal));
            Update_weights();
            for(int i=1;i<weights.Length;i++)
                parents[i-1].back_signal += back_signal * weights[i];
        }

    }
}
