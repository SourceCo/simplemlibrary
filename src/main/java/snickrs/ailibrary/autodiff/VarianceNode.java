package snickrs.ailibrary.autodiff;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
// i could've just used the library's capabilities already but i'm a masochist of course so here we are doing it ourselves
public class VarianceNode extends UnaryNode {
	public int[] axes;
	public INDArray mean;
	public boolean keepdims;
	public VarianceNode(Node left, boolean keepdims, int ... axes) {
		super(left);
		this.axes = axes;
		this.keepdims = keepdims;
	}
	public VarianceNode(Node left, int ... axes) {
		super(left);
		this.axes = axes;
		this.keepdims = true;
	}
	public VarianceNode(Node left) {
		super(left);
		this.axes = new int[] {-1};
		this.keepdims = true;
	}
	public INDArray child_evaluate() {
		this.mean = children.get(0).evaluate().mean(true, axes);
		this.m = Transforms.pow(children.get(0).m, 2.0f).mean(true, axes).sub(Transforms.pow(mean, 2.0f));
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		long[] shape = children.get(0).m.shape().clone();
		double inv_len = 2.0d;
		for(int axis : axes) {
			int idx = axis < 0 ? shape.length + axis : axis;
			inv_len /= shape[idx];
			shape[idx] = 1;
		}
		partials.set(0, upstream.reshape(shape).add(Nd4j.zeros(shape)).mul(children.get(0).m.sub(this.mean)).mul(inv_len));
	}
	public long[] shape() {
		long[] shape = children.get(0).shape().clone();
		for(int axis : axes) {
			shape[axis < 0 ? shape.length + axis : axis] = 1;
		}
		return shape;
	}
}
