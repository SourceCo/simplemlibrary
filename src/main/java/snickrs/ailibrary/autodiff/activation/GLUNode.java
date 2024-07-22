package snickrs.ailibrary.autodiff.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class GLUNode extends UnaryNode {
	public int axis;
	public long split;
	public INDArrayIndex[] a_indices;
	public INDArrayIndex[] b_indices;
	public INDArray a;
	public INDArray b;
	public GLUNode(Node left, int axis) {
		super(left);
		this.axis = axis;
		this.split = 0;
	}
	public GLUNode(Node left) {
		super(left);
		this.axis = -1;
		this.split = 0; // assume this because splitting the input to linear layers
	}
	public INDArray child_evaluate() {
		if(this.split == 0) {
			this.split = children.get(0).evaluate().size(axis)/2;
			long[] shape = children.get(0).m.shape();
			this.a_indices = new INDArrayIndex[shape.length];
			this.b_indices = new INDArrayIndex[shape.length];
			int t_axis = axis < 0 ? shape.length + axis : axis;
			for(int i = 0; i < shape.length; i++) {
				if(i == t_axis) {
					a_indices[i] = NDArrayIndex.interval(0, split);
					b_indices[i] = NDArrayIndex.interval(split, shape[i]);
				} else {
					a_indices[i] = NDArrayIndex.all();
					b_indices[i] = NDArrayIndex.all();
				}
			}
		} else {
			children.get(0).evaluate();
		}
		this.a = children.get(0).m.get(a_indices);
		this.b = Transforms.sigmoid(children.get(0).m.get(b_indices));
		this.m = a.mul(b);
		return this.m;
	}
	public void recalculate_dims() {
		long[] shape = children.get(0).shape();
		int t_axis = axis < 0 ? shape.length + axis : axis;
		this.split = shape[t_axis]/2;
		this.a_indices = new INDArrayIndex[shape.length];
		this.b_indices = new INDArrayIndex[shape.length];
		for(int i = 0; i < shape.length; i++) {
			if(i == t_axis) {
				a_indices[i] = NDArrayIndex.interval(0, split);
				b_indices[i] = NDArrayIndex.interval(split, shape[i]);
			} else {
				a_indices[i] = NDArrayIndex.all();
				b_indices[i] = NDArrayIndex.all();
			}
		}
	}
	public void child_diff(INDArray upstream) {
		// ideally multiply a and b in the same mul() call but they could have different dimensions when size along the axis is odd
		// for ex it could be 33 --> a = 16, b = 17
        INDArray grad = Nd4j.concat(axis, upstream.mul(b), upstream.mul(a).mul(b, b.rsub(1.0f)));
        partials.set(0, grad);
	}
	@Override
	public long[] shape() {
		long[] shape = children.get(0).shape().clone();
		int t_axis = axis < 0 ? shape.length + axis : axis;
		shape[t_axis] /= 2;
		return shape;
	}
}
