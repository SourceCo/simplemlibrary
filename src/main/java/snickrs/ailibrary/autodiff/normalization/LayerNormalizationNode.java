package snickrs.ailibrary.autodiff.normalization;

import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;
import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;
import snickrs.ailibrary.autodiff.weights.*;

public class LayerNormalizationNode extends Node {
	public float epsilon; // usually 0.001
	public boolean center; // whether to use beta (default true)
	public boolean scale; // whether to use gamma (default true)
	public boolean rms_scaling; // usually false
	public Node input;
	public Variable gamma;
	public Variable beta;
	public int[] axes;
	public LayerNormalizationNode(Node input) {
		super();
		init_vars(input, 0.001f, false, true, true, new int[] {-1});
	}
	public LayerNormalizationNode(Node input, float epsilon, boolean rms_scaling, boolean center, boolean scale, int ... axes) {
		super();
		init_vars(input, epsilon, rms_scaling, center, scale, axes);
	}
	public void recalculate_dims() {
		long[] shape = input.shape();
		long[] param_shape = new long[shape.length];
		Arrays.fill(param_shape, 1);
		for(int i = 0; i < axes.length; i++) {
			int val = axes[i] < 0 ? shape.length + axes[i] : axes[i];
			param_shape[val] = shape[val];
		}
		for(int i = 0; i < param_shape.length; i++) {
			if(gamma != null && param_shape[i] != gamma.dims[i]) gamma.expand_dims(i);
			if(beta != null && param_shape[i] != beta.dims[i]) beta.expand_dims(i);
		}
	}
	public void init_vars(Node input, float epsilon, boolean rms_scaling, boolean center, boolean scale, int ... axes) {
		this.axes = axes;
		this.input = input;
		this.rms_scaling = rms_scaling;
		this.epsilon = epsilon;
		this.center = center;
		this.scale = scale;
		long[] shape = input.shape();
		long[] param_shape = new long[shape.length];
		Arrays.fill(param_shape, 1);
		for(int i = 0; i < axes.length; i++) {
			int val = axes[i] < 0 ? shape.length + axes[i] : axes[i];
			param_shape[val] = shape[val];
		}
		if(this.scale || this.rms_scaling) this.gamma = new Variable("gamma", new Ones(), param_shape);
		if(this.center && !this.rms_scaling) this.beta = new Variable("beta", new Zeros(), param_shape);
		if(this.rms_scaling) {
			Node variance = new VarianceNode(input, this.axes);
			Node inv = new RsqrtNode(new AdditionNode(variance, new ConstantNode(this.epsilon)));
			addChild(new ProductNode(input, inv, this.gamma));
		} else {
			Node mean = new MeanNode(input, this.axes);
		    Node var = new VarianceNode(input, this.axes);
		    Node inv;
		    if(gamma == null) inv = new RsqrtNode(new AdditionNode(var, new ConstantNode(this.epsilon)));
		    else inv = new ProductNode(new RsqrtNode(new AdditionNode(var, new ConstantNode(this.epsilon))), gamma);
		    Node res;
		    if(beta == null) res = new ProductNode(new NegNode(mean), inv);
		    else res = new AdditionNode(new ProductNode(new NegNode(mean), inv), beta);
		    addChild(new AdditionNode(new ProductNode(input, inv), res));
		}
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
