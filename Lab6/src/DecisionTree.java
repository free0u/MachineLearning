import java.util.ArrayList;
import java.util.List;

/**
 * Created by free0u on 12/12/14.
 */
public class DecisionTree {
    private class Node {
        boolean isLeaf;
        int cl;

        int ruleInd, ruleValue;

        Node left, right;

        public Node(int ruleInd, int ruleValue, Node left, Node right) {
            this.cl = -1;
            this.ruleInd = ruleInd;
            this.ruleValue = ruleValue;
            this.left = left;
            this.right = right;
            this.isLeaf = false;
        }

        public Node(int cl) {
            this(-1, -1, null, null);
            this.isLeaf = true;
            this.cl = cl;
        }
    }

    private Node root;
    private int N, M;

    boolean IG;

    public DecisionTree(boolean IG) {
        this.IG = IG;
    }

    private int[] countClasses(List<Item> data) {
        int cnt0 = 0, cnt1 = 0;
        for (Item item : data) {
            if (item.cl == -1) {
                cnt0++;
            } else {
                cnt1++;
            }
        }
        return new int[]{cnt0, cnt1};
    }

    private List<List<Item>> split(List<Item> data, int ruleInd, int ruleValue) {
        ArrayList<Item> left = new ArrayList<>();
        ArrayList<Item> right = new ArrayList<>();

        for (Item item : data) {
            if (item.features.get(ruleInd) <= ruleValue) {
                left.add(item);
            } else {
                right.add(item);
            }
        }

        ArrayList<List<Item>> ret = new ArrayList<>();
        ret.add(left);
        ret.add(right);
        return ret;
    }

    private double qualitySet(List<Item> data) {
        int counts[] = countClasses(data);
        int count1 = counts[0];
        int count2 = counts[1];

        double p1 = (double) (count1) / (count1 + count2);
        double p2 = (double) (count2) / (count1 + count2);

        if (IG) {
            double ret = 0;
            if (count1 != 0) {
                ret -= p1 * logBin(p1);
            }
            if (count2 != 0) {
                ret -= p2 * logBin(p2);
            }
            return ret;
        } else {
            return 1 - (p1 * p1 + p2 * p2);
        }
    }

    private double logBin(double value) {
        return Math.log(value) / Math.log(2);
    }

    private double quality(List<Item> S, List<Item> left, List<Item> right) {
        double ret = qualitySet(S);
        ret -= (double) left.size() / S.size() * qualitySet(left);
        ret -= (double) right.size() / S.size() * qualitySet(right);

        return ret;
    }


    private int pruneQlt(Node v, List<Item> data) {
        List<Integer> predicted = testList(v, data);
        int cnt = 0;
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).cl != predicted.get(i)) {
                cnt++;
            }
        }
        return cnt;
    }

    private Node prune(Node v, List<Item> dataCurrent) {
        if (v.isLeaf) {
            return v;
        }

        // change to left child
        Node changeL = v.left;

        // change to right child
        Node changeR = v.right;

        // change to leaf 0
        Node changeLeaf0 = new Node(-1);

        // change to leaf 1
        Node changeLeaf1 = new Node(1);

        int cntBest = pruneQlt(v, dataCurrent);
        Node newNode = v;
        int t;
//        System.out.print(cntBest);

        t = pruneQlt(changeL, dataCurrent);
        if (t < cntBest) {
            cntBest = t;
            newNode = changeL;
        }
        t = pruneQlt(changeR, dataCurrent);
        if (t < cntBest) {
            cntBest = t;
            newNode = changeR;
        }
        t = pruneQlt(changeLeaf0, dataCurrent);
        if (t < cntBest) {
            cntBest = t;
            newNode = changeLeaf0;
        }
        t = pruneQlt(changeLeaf1, dataCurrent);
        if (t < cntBest) {
            cntBest = t;
            newNode = changeLeaf1;
        }

//        System.out.printf(" / %d\n", cntBest);

        if (newNode.isLeaf) {
            return newNode;
        }

        List<List<Item>> spl = split(dataCurrent, newNode.ruleInd, newNode.ruleValue);
        List<Item> left = spl.get(0);
        List<Item> right = spl.get(1);
        newNode.left = prune(newNode.left, left);
        newNode.right = prune(newNode.right, right);
        return newNode;

//        if (newNode == v) {
//            List<List<Item>> spl = split(dataCurrent, newNode.ruleInd, newNode.ruleValue);
//            List<Item> left = spl.get(0);
//            List<Item> right = spl.get(1);
//            newNode.left = prune(newNode.left, left);
//            newNode.right = prune(newNode.right, right);
//            return newNode;
//        }
//        return prune(newNode, dataCurrent);

//        return newNode;
    }

    private Node trainImpl(List<Item> data) {
        int counts[] = countClasses(data);
        if (counts[0] == 0 || counts[1] == 0) {
            Node node = new Node(counts[0] != 0 ? -1 : 1);
            return node;
        }

        Double bestQlt = null;
        int bestRuleInd = -1;
        int bestRuleValue = -1;

        int cnt = 0;
        for (Item item : data) {
            for (int j = 0; j < M; j++) {
                int ruleInd = j;
                int ruleValue = item.features.get(j);

                List<List<Item>> spl = split(data, ruleInd, ruleValue);
                List<Item> left = spl.get(0);
                List<Item> right = spl.get(1);

                if (left.size() == 0 || right.size() == 0) {
                    continue;
                }

                double qlt = quality(data, left, right);
                if (bestQlt == null) {
                    bestQlt = qlt;
                }
                if (qlt > bestQlt) {
                    bestQlt = qlt;
                    bestRuleInd = ruleInd;
                    bestRuleValue = ruleValue;
                }
            }
        }

        List<List<Item>> spl = split(data, bestRuleInd, bestRuleValue);
        List<Item> left = spl.get(0);
        List<Item> right = spl.get(1);


        Node node = new Node(bestRuleInd, bestRuleValue, trainImpl(left), trainImpl(right));

        return node;
    }

    private int treeH(Node v) {
        if (v.isLeaf) {
            return 1;
        }
        return (Math.max(treeH(v.left), treeH(v.right))) + 1;
    }

    public void train(List<Item> data) {
        N = data.size();

        int ind = (int) (N * 0.7);
        List<Item> dataTrain = data.subList(0, ind);
        List<Item> dataPrune = data.subList(ind, N - 1);

        N = dataTrain.size();
        M = dataTrain.get(0).features.size();

        root = trainImpl(dataTrain);
//        System.out.printf("height: %d\n", treeH(root));
        root = prune(root, dataPrune);
//        System.out.printf("height: %d\n", treeH(root));
    }

    private int testImpl(Node node, Item item) {
        if (node.isLeaf) {
            return node.cl;
        }
        int value = item.features.get(node.ruleInd);
        if (value <= node.ruleValue) {
            return testImpl(node.left, item);
        } else {
            return testImpl(node.right, item);
        }
    }

    private List<Integer> testList(Node v, List<Item> data) {
        List<Integer> ret = new ArrayList<>();
        for (Item item : data) {
            ret.add(testImpl(v, item));
        }
        return ret;
    }

    public List<Integer> test(List<Item> data) {
        return testList(root, data);
    }
}
