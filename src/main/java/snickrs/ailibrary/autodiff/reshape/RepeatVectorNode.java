package snickrs.ailibrary.autodiff.reshape;

import org.nd4j.linalg.api.ndarray.INDArray;
import snickrs.ailibrary.autodiff.*;

public class RepeatVectorNode extends UnaryNode {
    public long n;
    public RepeatVectorNode(Node left) {
        super(left);
        this.n = 1;
    }

    public RepeatVectorNode(Node left, long n) {
        super(left);
        this.n = n;
    }

    @Override
    public INDArray child_evaluate() {
    	long[] shape = children.get(0).evaluate().shape();
        this.m = children.get(0).m.reshape(new long[] {shape[0], 1, shape[1]}).repeat(1, n);
        return this.m;
    }

    @Override
    public void child_diff(INDArray upstream) {
        partials.set(0, unbroadcast(children.get(0).m, upstream));
    }

    @Override
    public long[] shape() {
    	long[] og = children.get(0).shape();
        return new long[] {og[0], n, og[1]};
    }
}