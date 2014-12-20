/**
 * Created by free0u on 12/20/14.
 */
public class Neuron {
    double[] w, d;
    double out;
    double grad;


    public Neuron(double[] w) {
        this.w = w;
        d = new double[w.length];
    }

    public double f(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double fd2(double x) {
        double ex = Math.exp(x);
        return ex / Math.pow(ex + 1, 2);
    }

    public double activate(double[] data) {
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            sum += data[i] * w[i];
        }
//        value += w[data.length];
        out = f(sum);
        return out;
    }
}
