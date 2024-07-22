package snickrs.ailibrary.autodiff.reshape;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import snickrs.ailibrary.autodiff.*;

public class CroppingNode extends UnaryNode {
    public int[][] crops;
    public INDArrayIndex[] indices;
    public CroppingNode(Node left, int[] ... crops) {
        super(left);
        this.crops = crops;
    }
    public CroppingNode(Node left, int ... crops) {
        super(left);
        this.crops = new int[crops.length][2];
        for(int i = 0; i < this.crops.length; i++) {
        	this.crops[i] = new int[] {crops[i], crops[i]};
        }
    }
    public void recalculate_dims() {
		long[] old_shape = children.get(0).m.shape();
		this.indices = new INDArrayIndex[old_shape.length];
		if(old_shape.length < crops.length+2) throw new IllegalArgumentException("received too many cropping dimensions");
		this.indices[0] = NDArrayIndex.all();
		this.indices[1] = NDArrayIndex.all();
		for(int i = 0; i < crops.length; i++) {
			this.indices[i+2] = NDArrayIndex.interval(crops[i][0], old_shape[i+2]-crops[i][1]);
		}
		for(int i = crops.length+2; i < old_shape.length; i++) {
			indices[i] = NDArrayIndex.all();
		}
    }
    @Override
    public INDArray child_evaluate() {
    	children.get(0).evaluate();
    	if(this.indices == null) {
    		long[] old_shape = children.get(0).m.shape();
    		this.indices = new INDArrayIndex[old_shape.length];
    		if(old_shape.length < crops.length+2) throw new IllegalArgumentException("received too many cropping dimensions");
    		this.indices[0] = NDArrayIndex.all();
    		this.indices[1] = NDArrayIndex.all();
    		for(int i = 0; i < crops.length; i++) {
    			this.indices[i+2] = NDArrayIndex.interval(crops[i][0], old_shape[i+2]-crops[i][1]);
    		}
    		for(int i = crops.length+2; i < old_shape.length; i++) {
    			indices[i] = NDArrayIndex.all();
    		}
    	}
        this.m = children.get(0).m.get(indices);
        return this.m;
    }

    @Override
    public void child_diff(INDArray upstream) {
    	INDArray grad = Nd4j.zerosLike(children.get(0).m);
    	grad.put(indices, upstream);
        partials.set(0, grad);
    }

    @Override
    public long[] shape() {
        long[] shape = children.get(0).shape().clone();
        for(int i = 0; i < crops.length; i++) {
        	shape[i+2] -= (crops[i][0]+crops[i][1]);
        }
        return shape;
    }
}