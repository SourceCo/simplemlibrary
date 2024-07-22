package snickrs.ailibrary.autodiff;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public abstract class Node {
	public List<Node> children; // child nodes
	public List<INDArray> partials; // partial derivatives with respect to each child
	public INDArray m;
	public INDArray gradient;
	public boolean training = true;
	public boolean trainable = true;
	public boolean evaluated = false; // don't wanna evaluate the same node twice either
	public boolean differentiated = false; // don't wanna call child_diff() on the same node twice
	public String name = "";
	public Node(Node ... children) {
		init(children);
	}
	public Node(List<? extends Node> children) {
		init(children);
	}
	public void init(Node ... children) {
		this.children = new ArrayList<Node>();
		this.partials = new ArrayList<INDArray>();
		for(Node child : children) {
			this.children.add(child);
			this.partials.add(Nd4j.scalar(0.0f));
		}
		this.m = Nd4j.scalar(0.0f);
		this.gradient = Nd4j.scalar(0.0f);
	}
	public void init(List<? extends Node> children) {
		this.children = new ArrayList<Node>();
		this.partials = new ArrayList<INDArray>();
		for(Node child : children) {
			this.children.add(child);
			this.partials.add(Nd4j.scalar(0.0f));
		}
		this.m = Nd4j.scalar(0.0f);
		this.gradient = Nd4j.scalar(0.0f);
	}
	public void addChild(Node child) {
		this.children.add(child);
		this.partials.add(Nd4j.scalar(0.0f));
	}
	public void setTraining(boolean training) {
		this.training = training;
		for(int i = 0; i < children.size(); i++) {
			children.get(i).setTraining(training);
		}
	}
	public void setEvaluated(boolean evaluated) {
		this.evaluated = evaluated;
		for(int i = 0; i < children.size(); i++) {
			children.get(i).setEvaluated(evaluated);
		}
	}
	public INDArray expand(INDArray arr, int ... axis) {
		long[] shape = arr.shape().clone();
		INDArrayIndex[] indices = new INDArrayIndex[shape.length];
		for(int i = 0; i < shape.length; i++) {
			indices[i] = NDArrayIndex.interval(0, shape[i]);
		}
		for(int x : axis) {
			shape[x < 0 ? shape.length + x : x] += 1;
		}
		INDArray new_arr = Nd4j.zeros(shape);
		new_arr.put(indices, arr);
		return new_arr;
	}
	public INDArray expand(INDArray arr, float value, int ... axis) {
		long[] shape = arr.shape().clone();
		INDArrayIndex[] indices = new INDArrayIndex[shape.length];
		for(int i = 0; i < shape.length; i++) {
			indices[i] = NDArrayIndex.interval(0, shape[i]);
		}
		for(int x : axis) {
			shape[x < 0 ? shape.length + x : x] += 1;
		}
		INDArray new_arr = Nd4j.zeros(shape).add(value);
		new_arr.put(indices, arr);
		return new_arr;
	}
	// expand to a certain shape, assuming adding one each time the expand method is called
	// not a general purpose method, no user of this library should even touch it unless
	// they know what they're doing
	public void update_dims() {
		recalculate_dims();
		for(Node child : children) {
			child.update_dims();
		}
	}
	public void recalculate_dims() {
		
	}
	public boolean isConstant(Node val) {
		return children.stream().allMatch(x -> x.isConstant(val));
	}
	public boolean isConstant() {
		return children.stream().allMatch(x -> x.isConstant());
	}
	public boolean isInteger() {
		return false;
	}
	public boolean equals(double val) {
		return false;
	}
	public INDArray evaluate() {
		if(!this.evaluated) {
			this.evaluated = true;
			return this.child_evaluate();
		}
		return this.m;
	}
	public INDArray child_evaluate() {
		return Nd4j.scalar(0.0f);
	}
	public ArrayList<INDArray> get_evals() {
		ArrayList<INDArray> evals = new ArrayList<INDArray>();
		for(Node child : children) {
			evals.add(child.evaluate());
		}
		return evals;
	}
	public INDArray[] get_evals_arr() {
		INDArray[] evals = new INDArray[children.size()];
		for(int i = 0; i < children.size(); i++) {
			evals[i] = children.get(i).evaluate();
		}
		return evals;
	}
	public ArrayList<Variable> get_parameters() {
		ArrayList<Variable> parameters = new ArrayList<Variable>();
		for(Node child : children) {
			if(child instanceof Variable) {
				parameters.add((Variable) child);
			} else {
				parameters.addAll(child.get_parameters());
			}
		}
		return parameters;
	}
	public void set(INDArray m) {
		this.m = m;
	}
	public void compute_gradients(INDArray seed) {
		this.child_diff(seed);
		for(int i = 0; i < children.size(); i++) {
			Node child = children.get(i);
			if(child.trainable) {
				child.gradient = child.gradient.add(partials.get(i));
				child.compute_gradients(partials.get(i));
			}
		}
	}
	public void compute_gradients() {
		compute_gradients(Nd4j.onesLike(this.m));
	}
	public void zero_grads() {
		this.gradient = Nd4j.scalar(0.0f);
		this.differentiated = false;
		this.evaluated = false;
		for(Node child : children) {
			child.zero_grads();
		}
	}
	public INDArray unbroadcast(long[] shape, INDArray g, int broadcast_idx) {
		while (g.shape().length > shape.length) {
			g = g.sum(broadcast_idx);
		}
		long[] tshape = shape.clone();
		if (g.shape().length == tshape.length) {
			for(int i = 0; i < tshape.length; i++) {
				if(tshape[i] == 1) {
					g = g.sum(true, i);
				}
			}
		}
		return g;
	}
	public INDArray unbroadcast(INDArray target, INDArray g, int broadcast_idx) {
		while (g.shape().length > target.shape().length) {
			g = g.sum(broadcast_idx);
		}
		long[] tshape = target.shape();
		if (g.shape().length == tshape.length) {
			for(int i = 0; i < tshape.length; i++) {
				if(tshape[i] == 1) {
					g = g.sum(true, i);
				}
			}
		}
		return g;
	}
	public INDArray unbroadcast(INDArray target, INDArray g) {
		return unbroadcast(target, g, 0);
	}
	public INDArray get_gradient(Node variable) {
		compute_gradients(Nd4j.onesLike(this.m));
		return variable.gradient;
	}
	public void diff(INDArray upstream) {
		if(differentiated) return;
		this.child_diff(upstream);
		this.differentiated = true;
	}
	public void child_diff(INDArray upstream) {
	}
	public boolean equals(Object a) {
		return this.hashCode() == a.hashCode();
	}
    protected int[] remove(int[] array, int index) {
        if (index < 0) return remove(array, array.length+index);
        if (index >= array.length) return array;
        int[] new_arr = new int[array.length - 1];
        for (int i = 0, j = 0; i < array.length; i++) {
            if (i == index) continue;
            new_arr[j++] = array[i];
        }
        return new_arr;
    }
    protected long[] remove(long[] array, long index) {
        if (index < 0) return remove(array, array.length+index);
        if (index >= array.length) return array;
        long[] new_arr = new long[array.length - 1];
        for (int i = 0, j = 0; i < array.length; i++) {
            if (i == index) continue;
            new_arr[j++] = array[i];
        }
        return new_arr;
    }
    protected int[] remove(int[] array, int ... index) {
    	int[] indices = friendly_indices(array, index);
    	if(indices.length >= array.length) throw new IllegalArgumentException("cannot remove more indices than there are elements in the array");
        int[] new_arr = new int[array.length - indices.length];
        for (int i = 0, j = 0; i < array.length; i++) {
            if (contains(indices, i)) continue;
            new_arr[j++] = array[i];
        }
        return new_arr;
    }
    protected long[] remove(long[] array, long ... index) {
    	long[] indices = friendly_indices(array, index);
    	if(indices.length >= array.length) throw new IllegalArgumentException("cannot remove more indices than there are elements in the array");
        long[] new_arr = new long[array.length - indices.length];
        for (int i = 0, j = 0; i < array.length; i++) {
            if (contains(indices, i)) continue;
            new_arr[j++] = array[i];
        }
        return new_arr;
    }
    protected long[] remove(long[] array, int ... index) {
    	long[] indices = friendly_indices(array, to_long(index));
    	if(indices.length >= array.length) throw new IllegalArgumentException("cannot remove more indices than there are elements in the array");
        long[] new_arr = new long[array.length - indices.length];
        for (int i = 0, j = 0; i < array.length; i++) {
            if (contains(indices, i)) continue;
            new_arr[j++] = array[i];
        }
        return new_arr;
    }
    protected long[] friendly_indices(long[] parent, long ... indices) {
    	long[] new_indices = indices.clone();
    	for(int i = 0; i < indices.length; i++) {
    		if(new_indices[i] < 0) new_indices[i] = parent.length+new_indices[i];
    	}
    	return new_indices;
    }
    protected int[] friendly_indices(int[] parent, int ... indices) {
    	int[] new_indices = indices.clone();
    	for(int i = 0; i < indices.length; i++) {
    		if(new_indices[i] < 0) new_indices[i] = parent.length+new_indices[i];
    	}
    	return new_indices;
    }
    protected boolean contains(int[] array, int element) {
    	for(int i = 0; i < array.length; i++) {
    		if(array[i] == element) return true;
    	}
    	return false;
    }
    protected boolean contains(long[] array, long element) {
    	for(int i = 0; i < array.length; i++) {
    		if(array[i] == element) return true;
    	}
    	return false;
    }
    protected boolean contains(Object[] array, Object element) {
    	for(int i = 0; i < array.length; i++) {
    		if(array[i].equals(element)) return true;
    	}
    	return false;
    }
    public int[] range(int start, int end) {
    	int[] ret = new int[end-start];
		for(int i = 0; i < ret.length; i++) {
			ret[i] = start+i;
		}
		return ret;
    }
	public long[] range(long[] arr) {
		long[] ret = new long[arr.length];
		for(int i = 0; i < arr.length; i++) {
			ret[i] = i;
		}
		return ret;
	}
    public long[] to_long(int[] arr) {
    	long[] indices = new long[arr.length];
    	for(int i = 0; i < arr.length; i++) {
    		indices[i] = (long) arr[i];
    	}
    	return indices;
    }
    public int[] to_int(long[] arr) {
    	int[] indices = new int[arr.length];
    	for(int i = 0; i < arr.length; i++) {
    		indices[i] = (int) arr[i];
    	}
    	return indices;
    }
    protected <T> T concat(T a, T b) {
        if (!a.getClass().isArray() || !b.getClass().isArray()) {
            throw new IllegalArgumentException();
        }
        Class<?> resCompType;
        Class<?> aCompType = a.getClass().getComponentType();
        Class<?> bCompType = b.getClass().getComponentType();
        if (aCompType.isAssignableFrom(bCompType)) {
            resCompType = aCompType;
        } else if (bCompType.isAssignableFrom(aCompType)) {
            resCompType = bCompType;
        } else {
            throw new IllegalArgumentException("types do not match, or are not assignable");
        }
        int aLen = Array.getLength(a);
        int bLen = Array.getLength(b);
        @SuppressWarnings("unchecked")
        T result = (T) Array.newInstance(resCompType, aLen + bLen);
        System.arraycopy(a, 0, result, 0, aLen);
        System.arraycopy(b, 0, result, aLen, bLen);        

        return result;
    }
	protected long[] broadcast(long[] shape1, long[] shape2) {
        int len1 = shape1.length;
        int len2 = shape2.length;
        int maxLen = Math.max(len1, len2);
        long[] resultShape = new long[maxLen];
        for (int i = 0; i < maxLen; i++) {
            long dim1 = i < len1 ? shape1[len1 - 1 - i] : 1;
            long dim2 = i < len2 ? shape2[len2 - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw new IllegalArgumentException("shapes are not broadcastable: " + java.util.Arrays.toString(shape1) + " and " + java.util.Arrays.toString(shape2));
            }
            resultShape[maxLen - 1 - i] = Math.max(dim1, dim2);
        }
        return resultShape;
    }
	public long[] shape() {
		return new long[] {1, 1};
	}
}
