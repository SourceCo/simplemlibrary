package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class UnaryNode extends Node {
	public UnaryNode(Node left) {
		super(left);
	}
	public boolean isConstant() {
		return children.get(0).isConstant();
	}
	public boolean isConstant(Variable val) {
		return children.get(0).isConstant(val);
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate();
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream);
	}
	public long[] shape() {
		return children.get(0).shape();
	}
}
