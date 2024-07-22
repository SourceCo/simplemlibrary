package snickrs.ailibrary.autodiff.dropout;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import snickrs.ailibrary.autodiff.*;
// only supports NCW, NCHW, or NCDHW --> equivalent to channels_first in keras
public class SpatialDropoutNode extends UnaryNode {
	public float rate; // rate of dropout
	public long[] mask_shape;
	public INDArray mask;
	public SpatialDropoutNode(Node left) {
		super(left);
		this.rate = 0.2f;
	}
	public SpatialDropoutNode(Node left, float rate) {
		super(left);
		this.rate = rate;
	}
	public void recalculate_dims() {
		long[] shape = children.get(0).shape();
		this.mask_shape = new long[shape.length];
		if(shape.length < 3) throw new IllegalArgumentException("input shape not supported for spatial dropout");
		Arrays.fill(mask_shape, 1);
		mask_shape[0] = shape[0]; // batch dimension
    	mask_shape[1] = shape[1]; // channel dimension
	}
	public INDArray child_evaluate() {
        if (training) {
        	if(this.mask_shape == null) {
        		long[] shape = children.get(0).evaluate().shape();
        		this.mask_shape = new long[shape.length];
        		if(shape.length < 3) throw new IllegalArgumentException("input shape not supported for spatial dropout");
        		Arrays.fill(mask_shape, 1);
        		mask_shape[0] = shape[0]; // batch dimension
            	mask_shape[1] = shape[1]; // channel dimension
                this.mask = Nd4j.randomBernoulli(1.0f-rate, mask_shape).divi(1.0f-rate);
                this.m = children.get(0).m.mul(this.mask);
        	} else {
                this.mask = Nd4j.randomBernoulli(1.0f-rate, mask_shape).divi(1.0f-rate);
                this.m = children.get(0).evaluate().mul(this.mask);
        	}
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
