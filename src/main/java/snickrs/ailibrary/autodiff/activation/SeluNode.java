package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.functions.Functions;

public class SeluNode extends UnaryNode {
	public float alpha;
	public float scale;
	public SeluNode(Node left) {
		super(left);
		alpha = 1.67326324f;
		scale = 1.05070098f;
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().dup();
		for(int i = 0; i < m.length(); i++) {
			m.putScalar(i, Functions.selu(m.getFloat(i)));
		}
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray deriv = Nd4j.zerosLike(this.m);
		for(int i = 0; i < m.length(); i++) {
			float x = this.children.get(0).m.getFloat(i);
			deriv.putScalar(i, x > 0.0f ? scale : scale * alpha * Math.exp(x));
		}
		partials.set(0, upstream.mul(deriv));
	}
}
