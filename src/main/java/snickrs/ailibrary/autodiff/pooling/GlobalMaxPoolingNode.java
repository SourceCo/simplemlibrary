package snickrs.ailibrary.autodiff.pooling;

import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;
// only supports NCW (1D), NCHW (2D), and NCDHW (3D) --> equivalent to channels first in keras
public class GlobalMaxPoolingNode extends UnaryNode {
	public boolean keepdims;
	public int[] axes;
	public GlobalMaxPoolingNode(Node left, boolean keepdims) {
		super(left);
		this.keepdims = keepdims;
	}
	public GlobalMaxPoolingNode(Node left) {
		super(left);
		this.keepdims = false;
	}
	public INDArray child_evaluate() {
		long[] shape = children.get(0).evaluate().shape();
		this.axes = range(2, shape.length);
		this.m = children.get(0).m.max(keepdims, axes);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		// because it's a max operation the arrays are always broadcastable -- provided we keep dimensions
		long[] new_shape = concat(this.m.shape(), ones(axes.length));
		partials.set(0, children.get(0).m.eq(keepdims ? this.m : this.m.reshape(new_shape)).castTo(Nd4j.defaultFloatingPointType()).mul(keepdims ? upstream : upstream.reshape(new_shape)));
	}
	public long[] ones(int len) {
		long[] res = new long[len];
		Arrays.fill(res, 1);
		return res;
	}
    @Override
    public long[] shape() {
        long[] input_shape = children.get(0).shape();
        if (keepdims) {
            long[] output_shape = input_shape.clone();
            for (int axis : axes) {
                output_shape[axis] = 1;
            }
            return output_shape;
        } else {
            long[] output_shape = new long[input_shape.length - axes.length];
            output_shape[0] = input_shape[0]; // batch dimension
            output_shape[1] = input_shape[1]; // num channels
            return output_shape;
        }
    }
}
