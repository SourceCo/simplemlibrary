package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class Log1pNode extends UnaryNode {
	public Log1pNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.log1p(children.get(0).evaluate(), true);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.div(children.get(0).m.add(1.0f)));
	}
}
