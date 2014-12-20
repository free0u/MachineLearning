import javax.swing.plaf.basic.BasicSpinnerUI;
import java.util.ArrayList;
import java.util.Arrays;
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
        double speed = 0.02;
        double alp = 0.5;

        for (Image image : data) {
            double[] outcome = testImpl(image);
            double[] rightOutcome = new double[10];
            rightOutcome[image.type] = 1.0;

            double err;

            // change out layer
            for (int i = 0; i < 10; i++) {
                err = rightOutcome[i] - outcome[i];

                Neuron neuron = outLayer.get(i);
                neuron.grad = err * neuron.fd(neuron.value);

                for (int j = 0; j < neuron.w.length; j++) {
                    double prevValue = 1;
                    if (j != neuron.w.length - 1) {
                        Neuron prevNeuron = hiddenLayer.get(j);
                        prevValue = prevNeuron.f(prevNeuron.value);
                    }
                    neuron.d[j] = speed * neuron.grad * prevValue;
                }
            }

            // change hidden layer
            for (int i = 0; i < hiddenLayer.size(); i++) {
                Neuron neuron = hiddenLayer.get(i);

                err = 0;
                for (int j = 0; j < 10; j++) {
                    Neuron nextNeuron = outLayer.get(j);
                    err += nextNeuron.grad * nextNeuron.w[i];
                }

                neuron.grad = err * neuron.fd(neuron.value);

                for (int j = 0; j < neuron.w.length; j++) {
                    double prevValue = 1;
                    if (j != neuron.w.length - 1) {
                        prevValue = image.data[j];
                    }
                    neuron.d[j] = speed * neuron.grad * prevValue;
                }

            }

            // change weight of out layer
            for (int i = 0; i < 10; i++) {
                Neuron neuron = outLayer.get(i);
                for (int j = 0; j < neuron.w.length; j++) {
                    neuron.w[j] += neuron.d[j];
                }
            }

            // change weight of hidden layer
            for (int i = 0; i < hiddenLayer.size(); i++) {
                Neuron neuron = hiddenLayer.get(i);
                for (int j = 0; j < neuron.w.length; j++) {
                    neuron.w[j] += neuron.d[j];
                }
            }


//            break;
        }
    }


    public double[] testImpl(Image ig) {
        double[] hiddenOutValues = new double[hiddenLayer.size()];
        for (int i = 0; i < hiddenLayer.size(); i++) {
            hiddenOutValues[i] = hiddenLayer.get(i).activate(ig.data);
        }

        double[] outOutValues = new double[10];
        for (int i = 0; i < outLayer.size(); i++) {
            outOutValues[i] = outLayer.get(i).activate(hiddenOutValues);
        }

        return outOutValues;
    }

    public byte test(Image ig) {
        double[] outValues = testImpl(ig);

        byte bestInd = 0;
        double bestValue = outValues[0];
        for (int i = 1; i < outValues.length; i++) {
            if (outValues[i] > bestValue) {
                bestInd = (byte) i;
            }
        }

        return bestInd;
    }

}
