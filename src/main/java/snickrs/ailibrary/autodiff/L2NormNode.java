package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class L2NormNode extends UnaryNode {
	public int axis;
	public L2NormNode(Node left, int axis) {
		super(left);
		this.axis = axis;
	}
	public L2NormNode(Node left) {
		super(left);
		this.axis = -1;
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().norm2(axis);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
        long[] shape = children.get(0).m.shape().clone();
        shape[axis < 0 ? shape.length + axis : axis] = 1;
        partials.set(0, upstream.reshape(shape).add(Nd4j.zeros(shape)).mul(children.get(0).m).div(this.m));
	}
}
