package snickrs.ailibrary.autodiff.dropout;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;

public class GaussianDropoutNode extends UnaryNode {
	public float rate; // rate of dropout
	public float stddev; // standard deviation
	public INDArray mask;
	public Distribution dist;
	public GaussianDropoutNode(Node left) {
		super(left);
		this.rate = 0.2f;
		initDist();
	}
	public GaussianDropoutNode(Node left, float rate) {
		super(left);
		this.rate = rate;
		initDist();
	}
	public void initDist() {
		this.stddev = (float) Math.sqrt(this.rate/(1.0f-this.rate));
		this.dist = Nd4j.getDistributions().createNormal(1.0f, this.stddev);
	}
	public INDArray child_evaluate() {
        if (training) {
            this.mask = Nd4j.rand(dist, children.get(0).evaluate().shape());
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
