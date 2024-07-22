package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;

public class NegNode extends UnaryNode {
	public NegNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().neg();
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.neg());
	}
}
