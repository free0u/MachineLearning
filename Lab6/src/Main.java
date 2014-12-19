import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by free0u on 12/12/14.
 */
public class Main {
    public final static int N = 100;
    public final static int M = 10000;

    public static double fScore(List<Integer> correct, List<Integer> predicted) {
        int tp = 0;
        int fp = 0;
        int fn = 0;

        int cnt = 0;
        for (int i = 0; i < correct.size(); i++) {
            int o = predicted.get(i);
            int c = correct.get(i);
            if (o == 1 && c == 1) {
                tp++;
            }
            if (o == 1 && c == -1) {
                fp++;
            }
            if (o == -1 && c == 1) {
                fn++;
            }
        }
        double f1 = 2.0 * tp / (2.0 * tp + fp + fn);
        return f1;
    }

    public static void main(String[] args) throws FileNotFoundException {
        Scanner fTrainData = new Scanner(new File("arcene_train.data"));
        Scanner fTrainLabel = new Scanner(new File("arcene_train.labels"));
        Scanner fValidData = new Scanner(new File("arcene_valid.data"));
        Scanner fValidLabel = new Scanner(new File("arcene_valid.labels"));


        System.out.println("> Read train");
        List<Item> trainItems = new ArrayList<>();
        List<Integer> trainLabel = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            List<Integer> f = new ArrayList<>();
            for (int j = 0; j < M; j++) {
                f.add(fTrainData.nextInt());
            }
            int label = fTrainLabel.nextInt();
            trainLabel.add(label);
            trainItems.add(new Item(f, label));
        }
        System.out.println("ok");

        System.out.println("> Read test");
        List<Item> validItems = new ArrayList<>();
        List<Integer> validLabel = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            List<Integer> f = new ArrayList<>();
            for (int j = 0; j < M; j++) {
                f.add(fValidData.nextInt());
            }
            int label = fValidLabel.nextInt();
            validLabel.add(label);
            validItems.add(new Item(f, label));
        }
        System.out.println("ok");


        // IG
        System.out.println("> Train tree IG");
        DecisionTree treeIG = new DecisionTree(true);
        treeIG.train(trainItems);
        System.out.println("ok");

        System.out.println("> Test tree IG");
        List<Integer> predictedLabels = treeIG.test(validItems);
        System.out.println("ok");

        System.out.println("> IG F1");
        System.out.println(fScore(validLabel, predictedLabels));

        // GIGI
        System.out.println("> Train tree GIGI");
        DecisionTree treeGIGI = new DecisionTree(false);
        treeGIGI.train(trainItems);
        System.out.println("ok");

        System.out.println("> Test tree GIGI");
        predictedLabels = treeGIGI.test(validItems);
        System.out.println("ok");

        System.out.println("> GIGI F1");
        System.out.println(fScore(validLabel, predictedLabels));
    }


}
