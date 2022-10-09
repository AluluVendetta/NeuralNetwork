using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_network
{
    internal class Activation_Functions
    {
        public static double Id(double x)
        {
            return x; 
        }
        public static double Id_derivative(double x)
        {
            return 1;
        }
        public static double Id_inverse(double x)
        {
            return x; 
        }
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        public static double Sigmoid_derivative(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x)); 
        }
        public static double Sigmoid_inverse(double x)
        {
            return Math.Log(x / (1.0 - x));
        }
    }
}
