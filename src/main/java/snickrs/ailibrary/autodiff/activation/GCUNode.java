package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class GCUNode extends UnaryNode {
	public INDArray cos_x;
	public GCUNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.cos_x = Transforms.cos(children.get(0).evaluate());
		this.m = children.get(0).m.mul(cos_x);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream.mul(cos_x.sub(children.get(0).m.mul(Transforms.sin(children.get(0).m)))));
	}
}
