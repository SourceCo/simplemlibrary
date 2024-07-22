package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class CosNode extends UnaryNode {
	public CosNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.cos(children.get(0).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(Transforms.sin(children.get(0).m)).neg());
	}
}

