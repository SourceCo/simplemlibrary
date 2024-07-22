package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import snickrs.ailibrary.autodiff.*;

public class SqrtNode extends UnaryNode {
	public SqrtNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.sqrt(children.get(0).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(0.5f).div(this.m));
	}
}

