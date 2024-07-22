package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ClipNode extends UnaryNode {
	public float min;
	public float max;
	public ClipNode(Node left, float min, float max) {
		super(left);
		this.min = min;
		this.max = max;
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.math().clipByValue(children.get(0).evaluate().dup(), min, max);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
        partials.set(0, upstream.mul(children.get(0).m.eq(this.m)));
	}
}
