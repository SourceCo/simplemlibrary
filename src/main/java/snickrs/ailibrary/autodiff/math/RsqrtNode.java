package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class RsqrtNode extends UnaryNode {
	public RsqrtNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.math().rsqrt(children.get(0).evaluate().dup());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(-0.5f).mul(Transforms.pow(this.m, 3.0f)));
	}
}

