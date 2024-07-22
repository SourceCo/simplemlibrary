package snickrs.ailibrary.autodiff.dropout;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;

public class GaussianNoiseNode extends UnaryNode {
	public float stddev; // standard deviation
	public INDArray mask;
	public Distribution dist;
	public GaussianNoiseNode(Node left) {
		super(left);
		this.stddev = 1.0f;
		initDist();
	}
	public GaussianNoiseNode(Node left, float stddev) {
		super(left);
		this.stddev = stddev;
		initDist();
	}
	public void initDist() {
		this.dist = Nd4j.getDistributions().createNormal(0.0f, this.stddev);
	}
	public INDArray child_evaluate() {
        if (training) {
            this.m = children.get(0).m.add(Nd4j.rand(dist, children.get(0).evaluate().shape()));
        } else {
            this.m = children.get(0).evaluate();
        }
        return this.m;
	}
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream);
	}
}
