package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MinNode extends Node {
	public MinNode(Node left, Node right) {
		super(left, right);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.min(children.get(0).evaluate(), children.get(1).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray deriv = children.get(0).m.lt(children.get(1).m);
		partials.set(0, unbroadcast(children.get(0).m, upstream.mul(deriv)));
		partials.set(1, unbroadcast(children.get(1).m, upstream.mul(deriv.rsub(1.0f))));
	}
    public long[] shape() {
    	return broadcast(children.get(0).shape(), children.get(1).shape());
    }
}
