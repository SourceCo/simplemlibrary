package snickrs.ailibrary.autodiff.merge;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import snickrs.ailibrary.autodiff.*;

public class ConcatenateNode extends Node {
    public int axis;
    public ConcatenateNode(int axis, Node ... children) {
        super(children);
        this.axis = axis;
    }
    public ConcatenateNode(Node ... children) {
        super(children);
        this.axis = -1;
    }
    public ConcatenateNode(int axis, List<Node> children) {
        super(children);
        this.axis = axis;
    }
    public ConcatenateNode(List<Node> children) {
        super(children);
        this.axis = -1;
    }

    @Override
    public INDArray child_evaluate() {
        this.m = Nd4j.concat(axis, get_evals_arr());
        return this.m;
    }

    @Override
    public void child_diff(INDArray upstream) {
    	long start = 0;
    	for(int i = 0; i < children.size(); i++) {
    		long amt = children.get(i).m.size(axis);
    		INDArrayIndex[] indices = new INDArrayIndex[children.get(i).m.shape().length];
    		for(int ind = 0; ind < indices.length; ind++) {
    			indices[ind] = (ind == axis || ind == indices.length+axis) ? NDArrayIndex.interval(start, start+amt) : NDArrayIndex.all();
    		}
    		partials.set(i, upstream.get(indices));
    		start += amt;
    	}
    }
    @Override
    public long[] shape() {
        long[] shape = children.get(0).shape().clone();
        for (int i = 1; i < children.size(); i++) {
            long[] child_shape = children.get(i).shape();
            for (int j = 0; j < shape.length; j++) {
                if (j == axis) {
                    shape[j] += child_shape[j];
                } else if (shape[j] != child_shape[j]) {
                    throw new IllegalArgumentException("All dimensions except the concatenation axis must match.");
                }
            }
        }
        return shape;
    }
}