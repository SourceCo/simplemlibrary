package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class SoftplusNode extends UnaryNode {
	public float alpha;
	public SoftplusNode(Node left, float alpha) {
		super(left);
		this.alpha = alpha;
	}
	public SoftplusNode(Node left) {
		super(left);
		this.alpha = 1.0f;
	}
	public INDArray child_evaluate() {
		this.m = alpha == 1.0f ? Transforms.softPlus(children.get(0).evaluate()) : Transforms.log(Transforms.exp(children.get(0).evaluate().mul(alpha)).add(1.0f)).div(alpha);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(alpha == 1.0f ? Transforms.sigmoid(children.get(0).m) : Transforms.sigmoid(children.get(0).m.mul(alpha))));
	}
}
