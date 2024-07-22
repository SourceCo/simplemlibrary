package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LTNode extends Node {
	public LTNode(Node left, Node right) {
		super(left, right);
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().lt(children.get(1).evaluate()).castTo(Nd4j.defaultFloatingPointType());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, unbroadcast(children.get(0).m, upstream.mul(this.m)));
		partials.set(1, unbroadcast(children.get(1).m, upstream.mul(this.m.rsub(1.0f))));
	}
    public long[] shape() {
    	return broadcast(children.get(0).shape(), children.get(1).shape());
    }
}
