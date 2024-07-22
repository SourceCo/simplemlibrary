package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class LReluNode extends UnaryNode {
	public double alpha;
	public LReluNode(Node left) {
		super(left);
		alpha = 0.2d;
	}
	public LReluNode(double alpha, Node left) {
		super(left);
		this.alpha = alpha;
	}
	public INDArray child_evaluate() {
		this.m = Transforms.leakyRelu(children.get(0).evaluate(), alpha);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(Transforms.leakyReluDerivative(children.get(0).m, alpha)));
	}
}
