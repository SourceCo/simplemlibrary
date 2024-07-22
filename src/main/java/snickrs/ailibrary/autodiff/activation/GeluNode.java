package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.functions.Functions;

public class GeluNode extends UnaryNode {
	public static final float PI = 3.1415927f;
	public boolean approximate;
	public GeluNode(Node left) {
		super(left);
		this.approximate = false;
	}
	public GeluNode(Node left, boolean approximate) {
		super(left);
		this.approximate = approximate;
	}
	public INDArray child_evaluate() {
		this.m = children.get(0).evaluate().dup();
		for(int i = 0; i < m.length(); i++) {
			m.putScalar(i, Functions.gelu(m.getFloat(i), approximate));
		}
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		INDArray deriv = Nd4j.zerosLike(this.m);
		for(int i = 0; i < m.length(); i++) {
			float x = this.children.get(0).m.getFloat(i);
			deriv.putScalar(i, approximate ? 0.5f + (0.398942f*x + 0.0535161f*Math.pow(x, 3.0f))*(1.0f-Math.pow(Math.tanh(0.797885f*x + 0.0356774f*Math.pow(x, 3.0f)), 2.0f)) + 0.5f*Math.tanh(0.797885f*x + 0.0356774f*Math.pow(x, 3.0f)) : Functions.p0(x) + x * Math.exp(-Math.pow(x, 2.0f)/2.0f) / Math.sqrt(2.0f * PI));
		}
		partials.set(0, upstream.mul(deriv));
	}
}
