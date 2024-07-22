package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MmulNode extends Node {
	public MmulNode(Node left, Node right) {
		super(left, right);
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().mmul(children.get(1).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mmul(children.get(1).m.transpose()));
		partials.set(1, children.get(0).m.transpose().mmul(upstream));
	}
	public long[] shape() {
		return new long[] {children.get(0).shape()[0], children.get(1).shape()[1]};
	}
}
