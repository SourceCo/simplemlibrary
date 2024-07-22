package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class SinhNode extends UnaryNode {
	public SinhNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.sinh(children.get(0).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(Transforms.cosh(children.get(0).m)));
	}
}

