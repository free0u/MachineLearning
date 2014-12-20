import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by free0u on 12/20/14.
 */
public class Main {
    public static List<Image> readImages(String name) throws IOException {
        List<Image> res = new ArrayList<>();

        byte[] rawLabels = Files.readAllBytes(Paths.get(name + ".labels"));
        byte[] rawData = Files.readAllBytes(Paths.get(name + ".data"));

        byte[] rawLen = Arrays.copyOfRange(rawLabels, 4, 8);
        ByteBuffer buf = ByteBuffer.wrap(rawLen);

        int countElements = buf.getInt();

        for (int i = 0; i < countElements; i++) {
            byte label = rawLabels[i + 8];

            int startInd = 16 + i * 28 * 28;
            int endInd = startInd + 28 * 28;

            byte[] image = Arrays.copyOfRange(rawData, startInd, endInd);

            res.add(new Image(image, label));
        }

        return res;
    }

    public static void main(String[] args) throws IOException {
        List<Image> trainImages = readImages("train");
        List<Image> testImages = readImages("test");

        NeuronNetwork net = new NeuronNetwork(30);
        net.train(trainImages);

        int cntErrors = 0;
        for (Image image : testImages) {
            int t = net.test(image);
            if (t != image.type) {
                cntErrors++;
            }
        }

        System.out.printf("Errors: %.02f%%\n", 100.0 * cntErrors / testImages.size());
    }
}
