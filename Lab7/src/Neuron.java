/**
 * Created by free0u on 12/20/14.
 */
public class Neuron {
    double[] w;

    public Neuron(double[] w) {
        this.w = w;
    }

    public double f(double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }

    public double activate(double[] data) {
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            sum += data[i] * w[i];
        }
        sum += w[data.length];
        return f(sum);
    }
}
