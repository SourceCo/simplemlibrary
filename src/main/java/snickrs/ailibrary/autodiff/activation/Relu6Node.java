package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class Relu6Node extends UnaryNode {
	public Relu6Node(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = Transforms.relu6(children.get(0).evaluate());
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray deriv = Nd4j.zerosLike(this.m);
		for(int i = 0; i < m.length(); i++) {
			float x = this.children.get(0).m.getFloat(i);
			deriv.putScalar(i, (x >= 0.0f && x <= 6.0f) ? 1.0f : 0.0f);
		}
		partials.set(0, upstream.mul(deriv));
	}
}
