package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class AtanNode extends UnaryNode {
	public AtanNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.atan(children.get(0).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.div(Transforms.pow(children.get(0).m, 2.0f).add(1.0f)));
	}
}

