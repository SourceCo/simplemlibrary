package snickrs.ailibrary.autodiff;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import snickrs.ailibrary.autodiff.constraints.*;
import snickrs.ailibrary.autodiff.weights.*;

public class Variable extends Node {
	public float updates;
	public long[] dims;
	public WeightInitScheme scheme;
	public List<Constraint> constraints;
	public Variable(String name, boolean trainable, WeightInitScheme scheme, long ... dims) {
		this.name = name;
		this.updates = 0.0f;
		this.scheme = scheme;
		this.dims = dims;
		if(this.scheme == null) this.m = Nd4j.rand(dims);
		else this.m = scheme.init(dims);
		this.constraints = new ArrayList<Constraint>();
		this.trainable = trainable;
	}
	public Variable(String name, boolean trainable, long ... dims) {
		this.name = name;
		this.updates = 0.0f;
		this.scheme = null;
		this.dims = dims;
		this.constraints = new ArrayList<Constraint>();
		this.m = Nd4j.rand(dims);
		this.trainable = trainable;
	}
	public Variable(String name, WeightInitScheme scheme, long ... dims) {
		this.name = name;
		this.updates = 0.0f;
		this.scheme = scheme;
		this.dims = dims;
		if(this.scheme == null) this.m = Nd4j.rand(dims);
		else this.m = scheme.init(dims);
		this.constraints = new ArrayList<Constraint>();
		this.trainable = true;
	}
	public Variable(String name, long ... dims) {
		this.name = name;
		this.updates = 0.0f;
		this.scheme = null;
		this.dims = dims;
		this.constraints = new ArrayList<Constraint>();
		this.m = Nd4j.rand(dims);
		this.trainable = true;
	}
	public Variable(String name, double val) { // only use with scalar inits
		this.name = name;
		this.updates = 0.0f;
		this.scheme = null;
		this.dims = new long[] {1};
		this.constraints = new ArrayList<Constraint>();
		this.m = Nd4j.scalar(val);
		this.trainable = true;
	}
	public Variable(String name) { // only use with scalar inits
		this.name = name;
		this.updates = 0.0f;
		this.scheme = null;
		this.dims = new long[] {1};
		this.constraints = new ArrayList<Constraint>();
		this.m = Nd4j.scalar(0.0f);
		this.trainable = true;
	}
	public void expand_dims(int ... axis) {
		long[] shape = this.m.shape().clone();
		INDArrayIndex[] indices = new INDArrayIndex[shape.length];
		for(int i = 0; i < shape.length; i++) {
			indices[i] = NDArrayIndex.interval(0, shape[i]);
		}
		for(int x : axis) {
			shape[x < 0 ? shape.length + x : x] += 1;
		}
		this.dims = shape;
		INDArray new_arr = this.scheme == null ? Nd4j.rand(shape) : scheme.init(shape);
		new_arr.put(indices, this.m);
		this.m = new_arr;
	}
	public Variable weightInit(WeightInitScheme scheme) {
		this.m = scheme.init(this.m.shape());
		return this;
	}
	public Variable addConstraint(Constraint c) {
		if(c != null) this.constraints.add(c);
		return this;
	}
	public void applyConstraints() {
		for(int i = 0; i < constraints.size(); i++) {
			constraints.get(i).apply(this.m);
		}
	}
	public boolean isConstant(Node val) {
		return val.hashCode() != this.hashCode();
	}
	@Override
	public String toString() {
		return "name: " + this.name + ", shape: " + Arrays.toString(this.dims);
	}
	@Override
	public INDArray child_evaluate() {
		return this.m;
	}
    public long[] shape() {
    	return this.dims;
    }
}