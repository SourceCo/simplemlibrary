package snickrs.ailibrary.autodiff.dropout;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;

public class AlphaDropoutNode extends UnaryNode {
	public float alpha = 1.67326324f;
    public float scale = 1.05070098f;
    public float alpha_p = -1.75809934f; // -alpha * scale
    public float a;
    public float b;
	public float rate; // rate of dropout
	public INDArray mask;
	public INDArray x;
	public AlphaDropoutNode(Node left) {
		super(left);
		this.rate = 0.2f;
		calcAB();
	}
	public AlphaDropoutNode(Node left, float rate) {
		super(left);
		this.rate = rate;
		calcAB();
	}
	public void calcAB() {
		// calculate affine transformation parameters - straight from keras implementation
		this.a = (float) Math.pow((1.0f-this.rate)*(1.0f+this.rate+Math.pow(alpha_p, 2.0f)), -0.5f);
		this.b = -a*alpha_p*this.rate;
	}
	public INDArray child_evaluate() {
        if (training) {
            this.mask = Nd4j.rand(children.get(0).evaluate().shape()).gt(rate).castTo(Nd4j.defaultFloatingPointType());
            this.x = children.get(0).m.mul(this.mask).add(mask.rsub(1.0f).mul(alpha_p));
            this.m = x.mul(a).add(b);
        } else {
            this.m = children.get(0).evaluate();
        }
        return this.m;
	}
	public void child_diff(INDArray upstream) {
		if(training) partials.set(0, upstream.mul(mask).mul(a));
		else partials.set(0, upstream);
	}
}
