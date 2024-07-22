package snickrs.ailibrary.autodiff.normalization;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;
import snickrs.ailibrary.autodiff.weights.*;

public class BatchNormalizationNode extends Node {
	public int axis; // usually -1
	public float momentum; // usually 0.99
	public float epsilon; // usually 0.001
	public boolean center; // whether to use beta (default true)
	public boolean scale; // whether to use gamma (default true)
	public Variable gamma;
	public Variable beta;
	public INDArray moving_mean;
	public INDArray moving_std;
	public int[] reduction_axes;
	public long[] shape;
	public Node input;
	public Node mean;
	public Node std;
	public BatchNormalizationNode(Node input) {
		super();
		init_vars(input, -1, 0.99f, 0.001f, true, true);
	}
	public BatchNormalizationNode(Node input, int axis, float momentum, float epsilon, boolean center, boolean scale) {
		super();
		init_vars(input, axis, momentum, epsilon, center, scale);
	}
	public void recalculate_dims() {
		this.shape = input.shape();
		int dim = axis < 0 ? shape.length + axis : axis;
		long[] param_shape = new long[shape.length];
	    Arrays.fill(param_shape, 1);
	    param_shape[dim] = shape[dim];
		for(int i = 0; i < param_shape.length; i++) {
			if(gamma != null && param_shape[i] != gamma.dims[i]) gamma.expand_dims(i);
			if(beta != null && param_shape[i] != beta.dims[i]) beta.expand_dims(i);
			if(param_shape[i] != moving_mean.size(i)) this.moving_mean = expand(this.moving_mean, i);
			if(param_shape[i] != moving_std.size(i)) this.moving_std = expand(this.moving_std, 1.0f, i);
		}
	}
	public void init_vars(Node input, int axis, float momentum, float epsilon, boolean center, boolean scale) {
		this.input = input;
		this.axis = axis;
		this.momentum = momentum;
		this.epsilon = epsilon;
		this.center = center;
		this.scale = scale;
		this.shape = input.shape();
		this.reduction_axes = remove(convert(shape), axis);
		int dim = axis < 0 ? shape.length + axis : axis;
		long[] param_shape = new long[shape.length];
	    Arrays.fill(param_shape, 1);
	    param_shape[dim] = shape[dim];
	    this.mean = new MeanNode(input, reduction_axes);
//	    this.std = new SqrtNode(new AdditionNode(new MeanNode(new SubtractionNode(new ExponentNode(input, new ConstantNode(2.0f)), new ExponentNode(new MeanNode(input, reduction_axes), new ConstantNode(2.0f))), reduction_axes), new ConstantNode(epsilon)));
	    this.std = new SqrtNode(new VarianceNode(input, reduction_axes));
	    Node norm = new QuotientNode(new SubtractionNode(input, mean), std);
		if(scale && center) {
			this.gamma = new Variable("gamma", new Ones(), param_shape);
			this.beta = new Variable("beta", new Zeros(), param_shape);
			addChild(new AdditionNode(new ProductNode(gamma, norm), beta));
		} else if(scale) {
			this.gamma = new Variable("gamma", new Ones(), param_shape);
			addChild(new ProductNode(gamma, norm));
		} else if(center) {
			this.beta = new Variable("beta", new Zeros(), param_shape);
			addChild(new AdditionNode(norm, beta));
		} else {
			addChild(norm);
		}
		this.moving_mean = Nd4j.zeros(param_shape);
		this.moving_std = Nd4j.ones(param_shape);
	}
	
    // convert to reduction axes
    public int[] convert(long[] arr) {
    	int[] indices = new int[arr.length];
    	for(int i = 0; i < arr.length; i++) {
    		indices[i] = i;
    	}
    	return indices;
    }
	public INDArray child_evaluate() {
		if(training) {
			this.m = children.get(0).evaluate();
			this.moving_mean.muli(this.momentum).addi(mean.m.mul(1.0f-this.momentum));
			this.moving_std.muli(this.momentum).addi(std.m.mul(1.0f-this.momentum));
		} else {
			INDArray x = this.input.evaluate();
			INDArray norm = x.sub(moving_mean).div(moving_std.add(epsilon));
			if(scale && center) {
				this.m = gamma.evaluate().mul(norm).add(beta.evaluate());
			} else if(scale) {
				this.m = gamma.evaluate().mul(norm);
			} else if(center) {
				this.m = norm.add(beta.evaluate());
			} else {
				this.m = norm;
			}
		}
		return this.m;
	}
	// too lazy to take derivatives myself so i'm gonna leave that to the child nodes
	public void child_diff(INDArray upstream) {
		partials.set(0, upstream);
	}
	public long[] shape() {
		return children.get(0).shape();
	}
}
