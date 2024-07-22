package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class AsinNode extends UnaryNode {
	public AsinNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.asin(children.get(0).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.div(Transforms.sqrt(Transforms.pow(children.get(0).m, 2.0f).rsub(1.0f))));
	}
}

