package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class LogSigmoidNode extends UnaryNode {
	public INDArray sigmoid_x;
	public LogSigmoidNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.sigmoid_x = Transforms.sigmoid(children.get(0).evaluate());
		this.m = Transforms.log(sigmoid_x);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(sigmoid_x.rsub(1.0f)));
	}
}
