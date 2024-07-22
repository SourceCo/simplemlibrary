package snickrs.ailibrary.autodiff.math;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class AcoshNode extends UnaryNode {
	public AcoshNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.math().acosh(children.get(0).evaluate().dup());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.div(Transforms.sqrt(Transforms.pow(children.get(0).m, 2.0f).sub(1.0f))));
	}
}

