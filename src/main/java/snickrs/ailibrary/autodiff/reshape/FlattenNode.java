package snickrs.ailibrary.autodiff.reshape;

import org.nd4j.linalg.api.ndarray.INDArray;
import snickrs.ailibrary.autodiff.*;

public class FlattenNode extends UnaryNode {
    public FlattenNode(Node left) {
        super(left);
    }

    @Override
    public INDArray child_evaluate() {
    	long[] old_shape = children.get(0).evaluate().shape();
        long flattened_dim = 1;
        if(old_shape.length > 1) {
        	for(int i = 1; i < old_shape.length; i++) {
        		flattened_dim *= old_shape[i];
        	}
        }
        // old_shape[0] is the batch size
        this.m = children.get(0).m.reshape(new long[] {old_shape[0], flattened_dim});
        return this.m;
    }

    @Override
    public void child_diff(INDArray upstream) {
        partials.set(0, upstream.reshape(children.get(0).m.shape()));
    }

    @Override
    public long[] shape() {
        long[] old_shape = children.get(0).shape();
        long flattened_dim = 1;
        if(old_shape.length > 1) {
        	for(int i = 1; i < old_shape.length; i++) {
        		flattened_dim *= old_shape[i];
        	}
        }
        // old_shape[0] is the batch size
        return new long[] {old_shape[0], flattened_dim};
    }
}