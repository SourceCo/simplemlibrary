package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class ErfNode extends UnaryNode {
	public static final float TWO_OVER_SQRT_PI = 1.12837917f;
	public ErfNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.math().erf(children.get(0).evaluate().dup());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(Transforms.exp(Transforms.neg(Transforms.pow(children.get(0).m, 2.0f)))).mul(TWO_OVER_SQRT_PI));
	}
}
