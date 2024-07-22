package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class LogSoftmaxNode extends UnaryNode {
	INDArray softmax_x;
	public LogSoftmaxNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.softmax_x = Transforms.softmax(children.get(0).evaluate());
		this.m = Transforms.log(softmax_x);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(softmax_x.rsub(1.0f)));
	}
}
