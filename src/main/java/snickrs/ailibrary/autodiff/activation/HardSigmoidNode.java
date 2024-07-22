package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.functions.Functions;

public class HardSigmoidNode extends UnaryNode {
	public static final float ONE_SIXTH = 0.16666667f;
	public HardSigmoidNode(Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().dup();
		for(int i = 0; i < m.length(); i++) {
			m.putScalar(i, Functions.hard_sigmoid(m.getFloat(i)));
		}
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray deriv = Nd4j.zerosLike(this.m);
		for(int i = 0; i < m.length(); i++) {
			float x = this.children.get(0).m.getFloat(i);
			deriv.putScalar(i, (x > -3.0f && x < 3.0f) ? ONE_SIXTH : 0.0f);
		}
		partials.set(0, upstream.mul(deriv));
	}
}
