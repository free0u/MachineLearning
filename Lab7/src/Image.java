import java.util.Arrays;

/**
 * Created by free0u on 12/20/14.
 */
public class Image {
    double[][] data;
    int type;

    public Image(byte[][] d, int type) {
        data = new double[d.length][d.length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data.length; j++) {
                byte t = d[i][j];
                data[i][j] = (double)t / 255;
            }
        }
        this.type = type;
    }

    @Override
    public String toString() {
        StringBuilder res = new StringBuilder();
        res.append(type);
        res.append("\n");
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data.length; j++) {
                res.append(data[i][j] < 0.5 ? 0 : 1);
            }
            res.append("\n");
        }
        return res.toString();
    }
}
