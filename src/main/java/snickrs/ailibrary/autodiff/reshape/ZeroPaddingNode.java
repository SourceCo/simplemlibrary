package snickrs.ailibrary.autodiff.reshape;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import snickrs.ailibrary.autodiff.*;

public class ZeroPaddingNode extends UnaryNode {
    public int[][] pads;
    public int[][] friendly_pads;
    public INDArrayIndex[] indices;
    public ZeroPaddingNode(Node left, int[] ... pads) {
        super(left);
        this.pads = pads;
        this.friendly_pads = new int[left.shape().length][2];
        friendly_pads[0] = new int[] {0, 0}; // preserve batch dimension
        friendly_pads[1] = new int[] {0, 0}; // preserve channel dimension
        for(int i = 2; i < friendly_pads.length; i++) {
        	if(i-2 >= this.pads.length) friendly_pads[i] = new int[] {0, 0};
        	else friendly_pads[i] = new int[] {this.pads[i-2][0], this.pads[i-2][1]};
        }
    }
    public ZeroPaddingNode(Node left, int ... pads) {
        super(left);
        this.pads = new int[pads.length][2];
        for(int i = 0; i < this.pads.length; i++) {
        	this.pads[i] = new int[] {pads[i], pads[i]};
        }
        this.friendly_pads = new int[left.shape().length][2];
        friendly_pads[0] = new int[] {0, 0}; // preserve batch dimension
        friendly_pads[1] = new int[] {0, 0}; // preserve channel dimension
        for(int i = 2; i < friendly_pads.length; i++) {
        	if(i-2 >= this.pads.length) friendly_pads[i] = new int[] {0, 0};
        	else friendly_pads[i] = new int[] {this.pads[i-2][0], this.pads[i-2][1]};
        }
    }
    public void recalculate_dims() {
        this.friendly_pads = new int[children.get(0).shape().length][2];
        friendly_pads[0] = new int[] {0, 0}; // preserve batch dimension
        friendly_pads[1] = new int[] {0, 0}; // preserve channel dimension
        for(int i = 2; i < friendly_pads.length; i++) {
        	if(i-2 >= this.pads.length) friendly_pads[i] = new int[] {0, 0};
        	else friendly_pads[i] = new int[] {this.pads[i-2][0], this.pads[i-2][1]};
        }
    }
    @Override
    public INDArray child_evaluate() {
    	// concat to keep the batch and channel dims
    	this.m = Nd4j.pad(children.get(0).m, friendly_pads, Pad.Mode.CONSTANT, 0.0d);
    	return this.m;
    }

    @Override
    public void child_diff(INDArray upstream) {
    	if(this.indices == null) {
    		long[] old_shape = this.m.shape();
    		this.indices = new INDArrayIndex[old_shape.length];
    		if(old_shape.length < pads.length+2) throw new IllegalArgumentException("received too many cropping dimensions");
    		this.indices[0] = NDArrayIndex.all();
    		this.indices[1] = NDArrayIndex.all();
    		for(int i = 0; i < pads.length; i++) {
    			this.indices[i+2] = NDArrayIndex.interval(pads[i][0], old_shape[i+2]-pads[i][1]);
    		}
    		for(int i = pads.length+2; i < old_shape.length; i++) {
    			indices[i] = NDArrayIndex.all();
    		}
    	}
        partials.set(0, upstream.get(indices));
    }

    @Override
    public long[] shape() {
        long[] shape = children.get(0).shape().clone();
        for(int i = 0; i < pads.length; i++) {
        	shape[i+2] += (pads[i][0]+pads[i][1]);
        }
        return shape;
    }
}