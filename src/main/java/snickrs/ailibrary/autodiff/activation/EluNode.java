package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.functions.Functions;

public class EluNode extends UnaryNode {
	public float alpha;
	public EluNode(Node left) {
		super(left);
		this.alpha = 1.0f;
	}
	public EluNode(double alpha, Node left) {
		super(left);
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().dup();
		for(int i = 0; i < m.length(); i++) {
			m.putScalar(i, Functions.elu(alpha, m.getFloat(i)));
		}
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray deriv = Nd4j.zerosLike(this.m);
		for(int i = 0; i < m.length(); i++) {
			float x = this.m.getFloat(i);
			deriv.putScalar(i, x >= 0.0f ? 1.0f : x+alpha);
		}
		partials.set(0, upstream.mul(deriv));
	}
}
