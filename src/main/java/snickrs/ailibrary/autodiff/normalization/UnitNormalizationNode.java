package snickrs.ailibrary.autodiff.normalization;

import org.nd4j.linalg.api.ndarray.INDArray;
import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.activation.SquareNode;
import snickrs.ailibrary.autodiff.math.*;

public class UnitNormalizationNode extends Node {
	public int[] axes;
	public Node input;
	public UnitNormalizationNode(Node input) {
		super();
		init_vars(input, new int[] {-1});
	}
	public UnitNormalizationNode(Node input, int ... axes) {
		super();
		init_vars(input, axes);
	}
	public void init_vars(Node input, int ... axes) {
		this.input = input;
		this.axes = axes;
		addChild(new ProductNode(this.input, new RsqrtNode(new SumNode(new SquareNode(this.input), this.axes))));
	}
    public INDArray child_evaluate() {
    	this.m = children.get(0).evaluate();
    	return this.m;
    }
    // too lazy to take derivatives myself so i'm gonna leave that to the child nodes
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream);
	}
}
