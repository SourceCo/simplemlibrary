package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class SoftmaxNode extends UnaryNode {
	public SoftmaxNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.softmax(children.get(0).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(this.m, this.m.rsub(1.0f)));
	}
}
