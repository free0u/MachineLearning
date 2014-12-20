/**
 * Created by free0u on 12/20/14.
 */
public class Image {
    double[] data;
    byte type;

    public Image(byte[] d, byte type) {
        data = new double[d.length];
        for (int i = 0; i < data.length; i++) {
            byte t = d[i];
            data[i] = (double) t / 255;
        }
        this.type = type;
    }

    @Override
    public String toString() {
        StringBuilder res = new StringBuilder();
        res.append(type);
        res.append("\n");
        for (int i = 0; i < data.length; i++) {
            res.append(data[i] < 0.5 ? 0 : 1);
            if (i % 28 == 0) {
                res.append("\n");
            }
        }
        return res.toString();
    }
}
