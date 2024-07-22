package snickrs.ailibrary.autodiff.dropout;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;

public class DropoutNode extends UnaryNode {
	public float rate; // rate of dropout
	public INDArray mask;
	public Distribution dist;
	public DropoutNode(Node left) {
		super(left);
		this.rate = 0.2f;
	}
	public DropoutNode(Node left, float rate) {
		super(left);
		this.rate = rate;
	}
	public INDArray child_evaluate() {
        if (training) {
            this.mask = Nd4j.rand(children.get(0).evaluate().shape()).gt(rate).castTo(Nd4j.defaultFloatingPointType()).div(1.0f-rate);
            this.m = children.get(0).m.mul(this.mask);
        } else {
            this.m = children.get(0).evaluate();
        }
        return this.m;
	}
	public void child_diff(INDArray upstream) {
		if(training) partials.set(0, upstream.mul(mask));
		else partials.set(0, upstream);
	}
}
