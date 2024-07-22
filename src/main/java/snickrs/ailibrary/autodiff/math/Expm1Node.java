package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class Expm1Node extends UnaryNode {
	public Expm1Node(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.expm1(children.get(0).evaluate(), true);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(this.m.add(1.0f)));
	}
}
