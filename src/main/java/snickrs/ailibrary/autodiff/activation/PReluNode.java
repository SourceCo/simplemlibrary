package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class PReluNode extends UnaryNode {
	public Variable alpha;
	public PReluNode(Node left) {
		super(left);
		this.alpha = new Variable("alpha", 0.0f);
		addChild(this.alpha);
	}
	public PReluNode(double alpha, Node left) {
		super(left);
		this.alpha = new Variable("alpha", alpha);
		addChild(this.alpha);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.leakyRelu(children.get(0).evaluate(), alpha.evaluate().getFloat(0));
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(Transforms.leakyReluDerivative(children.get(0).m, alpha.m.getFloat(0))));
		partials.set(1, Nd4j.scalar(upstream.mul(children.get(0).m.lt(0.0f)).sumNumber()));
	}
}
