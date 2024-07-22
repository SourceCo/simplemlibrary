package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class SoftsignNode extends UnaryNode {
	public SoftsignNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.softsign(children.get(0).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(Transforms.softsignDerivative(children.get(0).m)));
	}
}
