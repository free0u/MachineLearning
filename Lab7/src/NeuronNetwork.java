import javax.swing.plaf.basic.BasicSpinnerUI;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by free0u on 12/20/14.
 */
public class NeuronNetwork {
    List<Neuron> hiddenLayer, outLayer;
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

        hiddenLayer = new ArrayList<>();
        for (int i = 0; i < nHidden; i++) {
            Neuron neuron = new Neuron(genRandomDoubles(28 * 28 + 1));
            hiddenLayer.add(neuron);
        }


        outLayer = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Neuron neuron = new Neuron(genRandomDoubles(nHidden + 1));
            outLayer.add(neuron);
        }
    }

    public void train(List<Image> data) {

    }


    public byte test(Image ig) {
        double[] hiddenOutValues = new double[hiddenLayer.size()];
        for (int i = 0; i < hiddenLayer.size(); i++) {
            hiddenOutValues[i] = hiddenLayer.get(i).activate(ig.data);
        }

        double[] outOutValues = new double[10];
        for (int i = 0; i < outLayer.size(); i++) {
            outOutValues[i] = outLayer.get(i).activate(hiddenOutValues);
        }

        byte bestInd = 0;
        double bestValue = outOutValues[0];
        for (int i = 1; i < outOutValues.length; i++) {
            if (outOutValues[i] > bestValue) {
                bestInd = (byte) i;
            }
        }

        return bestInd;
    }

}
