package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class CbrtNode extends UnaryNode {
	public static final float ONE_THIRD = 0.3333333f;
	public CbrtNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.pow(children.get(0).evaluate(), ONE_THIRD);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(ONE_THIRD).div(Transforms.pow(this.m, 2.0f)));
	}
}

