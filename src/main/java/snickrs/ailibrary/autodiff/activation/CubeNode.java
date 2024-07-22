package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class CubeNode extends UnaryNode {
	public CubeNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.pow(children.get(0).evaluate(), 3.0f);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(Transforms.pow(children.get(0).m, 2.0f).mul(3.0f)));
	}
}
