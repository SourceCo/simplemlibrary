package snickrs.ailibrary.autodiff.reshape;

import org.nd4j.linalg.api.ndarray.INDArray;
import snickrs.ailibrary.autodiff.*;

public class ReshapeNode extends UnaryNode {
    public long[] new_shape;
    public ReshapeNode(Node left, long ... new_shape) {
        super(left);
        this.new_shape = new_shape;
    }

    @Override
    public INDArray child_evaluate() {
        this.m = children.get(0).evaluate().reshape(new_shape);
        return this.m;
    }

    @Override
    public void child_diff(INDArray upstream) {
        partials.set(0, upstream.reshape(children.get(0).m.shape()));
    }

    @Override
    public long[] shape() {
        return new_shape.clone();
    }
}