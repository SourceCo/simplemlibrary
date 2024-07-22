package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.functions.Functions;

public class RReluNode extends UnaryNode {
	public float lower;
	public float upper;
	public float avg;
	public RReluNode(Node left) {
		super(left);
		this.lower = 0.125f;
		this.upper = 0.3333333f;
		this.avg = (lower+upper)/2.0f;
	}
	public RReluNode(float lower, float upper, Node left) {
		super(left);
		this.lower = lower;
		this.upper = upper;
		this.avg = (lower+upper)/2.0f;
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().dup();
		for(int i = 0; i < m.length(); i++) {
			m.putScalar(i, Functions.rrelu(lower, upper, m.getFloat(i)));
		}
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray deriv = Nd4j.zerosLike(this.m);
		for(int i = 0; i < m.length(); i++) {
			float x = this.children.get(0).m.getFloat(i);
			deriv.putScalar(i, x <= 0.0f ? avg : 1.0f);
		}
		partials.set(0, upstream.mul(deriv));
	}
}
