/**
 * Created by free0u on 12/20/14.
 */
public class Neuron {
    double[] w, d;
    double value;
    double grad;


    public Neuron(double[] w) {
        this.w = w;
        d = new double[w.length];
    }

    public double f(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double fd(double x) {
        double ex = Math.exp(x);
        return ex / Math.pow(ex + 1, 2);
    }

    public double activate(double[] data) {
        value = 0;
        for (int i = 0; i < data.length; i++) {
            value += data[i] * w[i];
        }
        value += w[data.length];
        return f(value);
    }
}
