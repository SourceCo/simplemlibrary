package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class SiluNode extends UnaryNode {
	INDArray sigmoid_x;
	public float alpha;
	public SiluNode(float alpha, Node left) {
		super(left);
		this.alpha = alpha;
	}
	public SiluNode(Node left) {
		super(left);
		this.alpha = 1.0f;
	}
	public INDArray child_evaluate() {
		this.sigmoid_x = Transforms.sigmoid(children.get(0).evaluate().mul(alpha));
		this.m = children.get(0).m.mul(sigmoid_x);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(sigmoid_x.add(this.m.mul(alpha).mul(sigmoid_x, sigmoid_x.rsub(1.0f)))));
	}
}
