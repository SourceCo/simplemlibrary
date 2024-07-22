package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class TanhShrinkNode extends UnaryNode {
	INDArray tanh_x;
	public TanhShrinkNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.tanh_x = Transforms.tanh(children.get(0).evaluate());
		this.m = children.get(0).m.sub(tanh_x);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(Transforms.pow(tanh_x, 2.0f)));
	}
}
