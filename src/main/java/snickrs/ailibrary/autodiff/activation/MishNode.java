package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class MishNode extends UnaryNode {
	public INDArray tanh_softplus_x;
	public MishNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.tanh_softplus_x = Transforms.tanh(Transforms.softPlus(children.get(0).evaluate()));
		this.m = children.get(0).m.mul(tanh_softplus_x);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(
				tanh_softplus_x.add(children.get(0).m.mul(Transforms.sigmoid(children.get(0).m), Transforms.pow(tanh_softplus_x, 2.0f).rsub(1.0f)))));
	}
}
