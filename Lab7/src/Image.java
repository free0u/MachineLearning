/**
 * Created by free0u on 12/20/14.
 */
public class Image {
    double[] data;
    byte type;

    public Image(byte[] d, byte type) {
        data = new double[d.length];
        for (int i = 0; i < data.length; i++) {
            int t = d[i] & 0xFF;
            data[i] = (double) t / 255;
        }
        this.type = type;
    }

    @Override
    public String toString() {
        StringBuilder res = new StringBuilder();
        res.append(type);
        res.append("\n");
        int r = 0;
        for (int i = 0; i < data.length; i++) {
//            res.append(data[i] < 0.5 ? 0 : 1);
            int t = (int)(data[i] * 10) % 10;
            res.append(t == 0 ? " " : t);
            r++;
            if (r == 28) {
                res.append("\n");
                r = 0;
            }
        }
        return res.toString();
    }
}
