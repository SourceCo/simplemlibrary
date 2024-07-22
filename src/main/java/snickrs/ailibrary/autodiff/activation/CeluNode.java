package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.functions.Functions;

public class CeluNode extends UnaryNode {
	public float alpha;
	public CeluNode(Node left) {
		super(left);
		alpha = 0.2f;
	}
	public CeluNode(float alpha, Node left) {
		super(left);
		this.alpha = alpha;
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().dup();
		for(int i = 0; i < m.length(); i++) {
			m.putScalar(i, Functions.celu(alpha, m.getFloat(i)));
		}
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray deriv = Nd4j.zerosLike(this.m);
		for(int i = 0; i < m.length(); i++) {
			float x = this.children.get(0).m.getFloat(i);
			deriv.putScalar(i, x < 0.0f ? Math.exp(x/alpha) : 1.0f);
		}
		partials.set(0, upstream.mul(deriv));
	}
}
