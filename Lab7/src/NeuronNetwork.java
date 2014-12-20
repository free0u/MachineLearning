import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by free0u on 12/20/14.
 */
public class NeuronNetwork {
    List<List<Neuron>> net;
    Random rand;


    private double genSmallDouble(double d) {
        double res = rand.nextDouble() * 2;
        res -= 1;
        res *= d;
        return res;
    }

    private double[] genRandomDoubles(int n) {
        double[] res = new double[n];
        for (int i = 0; i < n; i++) {
            res[i] = genSmallDouble(0.02);
        }
        return res;
    }

    public NeuronNetwork(int nHidden) {
        rand = new Random(0);

        net = new ArrayList<>();

        List<Neuron> hidden = new ArrayList<>();
        for (int i = 0; i < nHidden; i++) {
            Neuron neuron = new Neuron(genRandomDoubles(28 * 28));
            hidden.add(neuron);
        }

        net.add(hidden);

        List<Neuron> out = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Neuron neuron = new Neuron(genRandomDoubles(nHidden));
            out.add(neuron);
        }

        net.add(out);
    }

    public void train(List<Image> data) {

    }


    public byte test(Image ig) {

        return 0;
    }

}
